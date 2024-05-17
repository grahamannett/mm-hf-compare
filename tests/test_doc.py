import unittest
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import torch

from tests.fixtures.get_fixtures import fixtures_data


class TestPaliGemmaModels(unittest.TestCase):
    def test_infer(self):
        model_id = "google/paligemma-3b-mix-448"
        image_fixtures = fixtures_data["fixtures"]["images"]
        image_fixture = next(filter(lambda x: x["name"] == "puffin", image_fixtures))
        image_path, image_prompt = image_fixture["path"], image_fixture["prompt"]
        prompt = f"Segment `${image_prompt}`"

        # should be
        assert "tests/fixtures/images/puffin.png" == image_path
        assert "detect puffin" in prompt
        image = Image.open(image_path)

        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        processor = PaliGemmaProcessor.from_pretrained(model_id)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        predictions = model.generate(**inputs, max_new_tokens=50)
        result = processor.decode(predictions[0], skip_special_tokens=True)

    @unittest.skip("Not implemented")
    def test_infer_diagram(self):
        model_id = "google/paligemma-3b-ft-ai2d-448"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        predictions = self.model_diagram.generate(**inputs, max_new_tokens=100)
        result = processor.decode(predictions[0], skip_special_tokens=True)[len(prompt) :].lstrip("\n")


if __name__ == "__main__":
    unittest.main()
