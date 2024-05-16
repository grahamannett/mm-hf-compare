#!/bin/bash

set -e
# set -x
FIXTURES_DIR="$(realpath $(dirname "$0"))"

PUFFIN_IMAGE_URL=("https://google-paligemma.hf.space/file=/tmp/gradio/78f93b49088f8d72ee546d656387403d647b413f/image.png" "puffin.png")
BBOX_IMAGE_URL=("https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg" "bbox.jpg")

IMAGE_FIXTURES_DIR="$FIXTURES_DIR/images"

echo -e "Images Dir [$IMAGE_FIXTURES_DIR]"

mkdir -p $IMAGE_FIXTURES_DIR

wget -O $IMAGE_FIXTURES_DIR/${PUFFIN_IMAGE_URL[1]} ${PUFFIN_IMAGE_URL[0]}
wget -O $IMAGE_FIXTURES_DIR/${BBOX_IMAGE_URL[1]} ${BBOX_IMAGE_URL[0]}

# MARK: External Fixtures
VAE_OID_URL=("https://huggingface.co/spaces/google/paligemma-hf/resolve/main/vae-oid.npz?download=true" "vae-oid.npz")

EXTERNAL_FIXTURES_DIR="$FIXTURES_DIR/external"

echo -e "External Dir [$EXTERNAL_FIXTURES_DIR]"
mkdir -p $EXTERNAL_FIXTURES_DIR

wget -O $EXTERNAL_FIXTURES_DIR/${VAE_OID_URL[1]} ${VAE_OID_URL[0]}
