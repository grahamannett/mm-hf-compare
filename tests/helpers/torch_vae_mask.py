import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tests.fixtures.get_fixtures import fixtures_data


"""NOTE:
Started moving the original code to torch, not entirely sure if it is fully
working/correct but it is really not that important.  Original file is:

https://huggingface.co/spaces/google/paligemma-hf/blob/main/app.py


"""

# "external/vae-oid.npz"
_VAE_PARAMS_PATH = fixtures_data["external_fixtures_dir"] / fixtures_data["fixtures"]["vae"]["filename"]


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to PyTorch params."""

    def transp(kernel):
        # dont transpose since we are using pytorch
        return kernel

    def conv(name):
        return {
            "bias": checkpoint[name + ".bias"],
            "weight": transp(checkpoint[name + ".weight"]),
        }

    def resblock(name):
        return {
            "Conv_0": conv(name + ".0"),
            "Conv_1": conv(name + ".2"),
            "Conv_2": conv(name + ".4"),
        }

    return (
        checkpoint["_vq_vae._embedding"],
        {
            "Conv_0": conv("decoder.0"),
            "ResBlock_0": resblock("decoder.2.net"),
            "ResBlock_1": resblock("decoder.3.net"),
            "ConvTranspose_0": conv("decoder.4"),
            "ConvTranspose_1": conv("decoder.6"),
            "ConvTranspose_2": conv("decoder.8"),
            "ConvTranspose_3": conv("decoder.10"),
            "Conv_1": conv("decoder.12"),
        },
    )


def convert_params_to_pytorch(custom_params, prefix="", state={}):
    for name, params in custom_params.items():
        if name in ["weight", "bias"]:
            state[f"{prefix}.{name}"] = torch.from_numpy(params)
        else:
            convert_params_to_pytorch(
                params,
                prefix=f"{prefix}.{name}" if prefix else name,
                state=state,
            )
    return state


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = embeddings[codebook_indices.reshape(-1)]
    encodings = encodings.reshape(batch_size, 4, 4, embedding_dim)
    return encodings


class ResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.Conv_0 = nn.Conv2d(features, features, 3, padding=1)
        self.Conv_1 = nn.Conv2d(features, features, 3, padding=1)
        self.Conv_2 = nn.Conv2d(features, features, 1)

    def forward(self, x):
        original_x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x + original_x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_res_blocks = 2
        num_upsample_layers = 4
        dim = 128

        self.Conv_0 = nn.Conv2d(dim * 4, dim, 1, padding=0)
        self.ResBlock_0 = ResBlock(dim)
        self.ResBlock_1 = ResBlock(dim)

        updim = dim
        downdim = dim

        self.ConvTranspose_0 = nn.ConvTranspose2d(updim, updim, 4, stride=2, padding=2)
        updim = downdim
        downdim = downdim // 2

        self.ConvTranspose_1 = nn.ConvTranspose2d(updim, downdim, 4, stride=2, padding=2)
        updim = downdim
        downdim = downdim // 2

        self.ConvTranspose_2 = nn.ConvTranspose2d(updim, downdim, 4, stride=2, padding=2)
        updim = downdim
        downdim = downdim // 2

        self.ConvTranspose_3 = nn.ConvTranspose2d(updim, downdim, 4, stride=2, padding=2)

        self.Conv_1 = nn.Conv2d(downdim, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for res_block in [self.ResBlock_0, self.ResBlock_1]:
            x = res_block(x)

        for upsample_layer in [getattr(self, f"ConvTranspose_{i}") for i in range(4)]:
            x = F.relu(upsample_layer(x))

        x = self.Conv_1(x)
        return x


def reconstruct_masks(codebook_indices, embeddings, decoder):
    quantized = _quantized_values_from_codebook_indices(codebook_indices, embeddings)
    quantized = torch.from_numpy(quantized).float()
    return decoder(quantized)


@functools.cache
def _get_reconstruct_mask():
    with open(_VAE_PARAMS_PATH, "rb") as f:
        embeddings, params = _get_params(dict(np.load(f)))

    params = convert_params_to_pytorch(params)
    decoder.load_state_dict(params)
    return functools.partial(reconstruct_masks, embeddings=embeddings, decoder=decoder)


if __name__ == "__main__":
    decoder = Decoder()

    seg_arr = np.array(list(range(1, 17)), dtype=np.int32)
    seg_arr = torch.tensor(list(range(16), dtype=torch.int32))[None, ...]
    out = _get_reconstruct_mask()(seg_arr)
