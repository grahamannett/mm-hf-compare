import requests
from pathlib import Path

import tomllib

# Define directories
script_dir = Path(__file__).parent
with open(script_dir / "fixtures.toml", "rb") as f:
    fixtures_data = tomllib.load(f)

image_fixtures_dir = script_dir / fixtures_data["image_fixtures_dir"]
external_fixtures_dir = script_dir / fixtures_data["external_fixtures_dir"]

# Ensure directories exist
image_fixtures_dir.mkdir(parents=True, exist_ok=True)
external_fixtures_dir.mkdir(parents=True, exist_ok=True)


# Function to download and save a file
def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the response was an error
    with open(filename, "wb") as f:
        f.write(response.content)


for image_fixture in fixtures_data["fixtures"]["images"]:
    image_name = image_fixture["name"]
    image_url = image_fixture["url"]
    filename = image_fixtures_dir / image_fixture["filename"]

    if filename.exists():
        print(f"Skipping download of [{filename}]")
        continue

    print(f"Downloading [{image_url}] to [{filename}]")
    download_file(image_url, filename)


print(f"Images Dir [{image_fixtures_dir}]")
print(f"External Dir [{external_fixtures_dir}]")
