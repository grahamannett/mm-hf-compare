import unittest

import torch
import PIL.Image, PIL.ImageDraw

import transformers
import numpy as np
from pathlib import Path

import re
from tests.helpers.vae_mask import _get_reconstruct_masks
import torch.nn as nn
import functools


COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]

fixtures_dir = Path(__file__).parent / "fixtures"


def infer(image: PIL.Image.Image, text: str, max_new_tokens: int, model, processor) -> str:
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return result[0][len(text) :].lstrip("\n")


_SEGMENT_DETECT_RE = re.compile(
    r"(.*?)" + r"<loc(\d{4})>" * 4 + r"\s*" + "(?:%s)?" % (r"<seg(\d{3})>" * 16) + r"\s*([^;<>]+)? ?(?:; )?",
)


def extract_objs(text, width, height, unique_labels=False):
    """Returns objs for a string with "<loc>" and "<seg>" tokens."""
    objs = []
    seen = set()
    while text:
        m = _SEGMENT_DETECT_RE.match(text)
        if not m:
            break
        print("m", m)
        gs = list(m.groups())
        before = gs.pop(0)
        name = gs.pop()

        ys, xs = [gs[0], gs[2]], [gs[1], gs[3]]
        xs = [round((int(x) / 1024) * width) for x in xs]
        ys = [round((int(y) / 1024) * height) for y in ys]

        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]

        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))

        seg_indices = gs[4:20]
        assert y1 == ys[0] and x1 == xs[0] and y2 == ys[1] and x2 == xs[1], "mismatch"

        if seg_indices[0] is None:
            mask = None
        else:
            seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
            (m64,) = _get_reconstruct_masks()(seg_indices[None])[..., 0]
            m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
            m64 = PIL.Image.fromarray((m64 * 255).astype("uint8"))
            mask = np.zeros([height, width])
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0

        content = m.group()
        if before:
            objs.append(dict(content=before))
            content = content[len(before) :]
        while unique_labels and name in seen:
            name = (name or "") + "'"
        seen.add(name)
        objs.append(dict(content=content, xyxy=(x1, y1, x2, y2), mask=mask, name=name))
        text = text[len(before) + len(content) :]

    if text:
        objs.append(dict(content=text))

    return objs


def parse_segmentation(input_image, input_text, model, processor):
    out = infer(input_image, input_text, max_new_tokens=100, model=model, processor=processor)
    objs = extract_objs(out.lstrip("\n"), input_image.size[0], input_image.size[1], unique_labels=True)
    labels = set(obj.get("name") for obj in objs if obj.get("name"))
    color_map = {l: COLORS[i % len(COLORS)] for i, l in enumerate(labels)}
    highlighted_text = [(obj["content"], obj.get("name")) for obj in objs]
    annotated_img = (
        input_image,
        [
            (
                obj["mask"] if obj.get("mask") is not None else obj["xyxy"],
                obj["name"] or "",
            )
            for obj in objs
            if "mask" in obj or "xyxy" in obj
        ],
    )
    has_annotations = bool(annotated_img[1])
    return annotated_img, color_map, has_annotations, highlighted_text


class TestSegment(unittest.TestCase):
    def test_segment(self):
        # from https://huggingface.co/spaces/google/paligemma-hf/blob/main/examples/cc_puffin.json

        model_id = "google/paligemma-3b-mix-448"
        image_path = "tests/fixtures/images/puffin.png"
        prompt = "detect puffin in the back; puffin in front"
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        processor = transformers.PaliGemmaProcessor.from_pretrained(model_id)

        model = model.eval()

        image = PIL.Image.open(image_path)
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        annotated_img, color_map, has_annotations, highlighted_text = parse_segmentation(
            image, prompt, model, processor
        )
        draw = PIL.ImageDraw.Draw(annotated_img[0])
        for obj in annotated_img[1]:
            if isinstance(obj[0], tuple):
                draw.rectangle(obj[0], outline=color_map[obj[1]], width=2)
            else:
                draw.rectangle(obj[0], fill=color_map[obj[1]], outline=color_map[obj[1]], width=2)


if __name__ == "__main__":
    unittest.main()
