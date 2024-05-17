import flax.linen as nn
import jax
import jax.numpy as jnp
import functools
import numpy as np

from tests.fixtures.get_fixtures import fixtures_data

# "external/vae-oid.npz"
_VAE_PARAMS_PATH = fixtures_data["external_fixtures_dir"] / fixtures_data["fixtures"]["vae"]["filename"]


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to Flax params."""

    def transp(kernel):
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        return {
            "bias": checkpoint[name + ".bias"],
            "kernel": transp(checkpoint[name + ".weight"]),
        }

    def resblock(name):
        return {
            "Conv_0": conv(name + ".0"),
            "Conv_1": conv(name + ".2"),
            "Conv_2": conv(name + ".4"),
        }

    return {
        "embeddings_alt": checkpoint["_vq_vae._embedding"],
        "_embeddings": checkpoint["_vq_vae._embedding"],
        "Conv_0": conv("decoder.0"),
        "ResBlock_0": resblock("decoder.2.net"),
        "ResBlock_1": resblock("decoder.3.net"),
        "ConvTranspose_0": conv("decoder.4"),
        "ConvTranspose_1": conv("decoder.6"),
        "ConvTranspose_2": conv("decoder.8"),
        "ConvTranspose_3": conv("decoder.10"),
        "Conv_1": conv("decoder.12"),
    }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings


@functools.cache
def _get_reconstruct_masks():
    """Reconstructs masks from codebook indices.
    Returns:
      A function that expects indices shaped `[B, 16]` of dtype int32, each
      ranging from 0 to 127 (inclusive), and that returns a decoded masks sized
      `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
    """

    class ResBlock(nn.Module):
        features: int

        @nn.compact
        def __call__(self, x):
            original_x = x
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
            return x + original_x

    class Decoder(nn.Module):
        """Upscales quantized vectors to mask."""

        @nn.compact
        def __call__(self, x):
            num_res_blocks = 2
            dim = 128
            num_upsample_layers = 4

            x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
            x = nn.relu(x)

            for _ in range(num_res_blocks):
                x = ResBlock(features=dim)(x)

            for _ in range(num_upsample_layers):
                x = nn.ConvTranspose(
                    features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding=2,
                    transpose_kernel=True,
                )(x)
                x = nn.relu(x)
                dim //= 2

            x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

            return x

    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(codebook_indices, params["_embeddings"])
        decoder = Decoder()

        return decoder.apply({"params": params}, quantized)

    with open(_VAE_PARAMS_PATH, "rb") as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend="cpu")


if __name__ == "__main__":
    seg_arr = np.array(list(range(1, 17)), dtype=np.int32)
    out = _get_reconstruct_masks()(seg_arr[None])[..., 0]
