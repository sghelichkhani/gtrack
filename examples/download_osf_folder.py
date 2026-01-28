#!/usr/bin/env python3
"""Download a specific folder from OSF recursively."""

import argparse
import json
import os
import urllib.request
from pathlib import Path


def fetch_json(url):
    """Fetch JSON from OSF API."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {dest_path.name}")
    urllib.request.urlretrieve(url, dest_path)


def download_folder(api_url, local_dir):
    """Recursively download folder contents from OSF."""
    data = fetch_json(api_url)

    for item in data.get("data", []):
        attrs = item["attributes"]
        name = attrs["name"]
        kind = attrs["kind"]

        if kind == "file":
            download_url = item["links"]["download"]
            dest_path = local_dir / name
            download_file(download_url, dest_path)
        elif kind == "folder":
            # Get the folder contents URL
            folder_url = item["relationships"]["files"]["links"]["related"]["href"]
            print(f"Entering folder: {name}/")
            download_folder(folder_url, local_dir / name)


def main():
    parser = argparse.ArgumentParser(description="Download OSF folder")
    parser.add_argument("--project", "-p", required=True, help="OSF project ID")
    parser.add_argument("--folder-id", "-f", required=True, help="OSF folder ID")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    args = parser.parse_args()

    api_url = f"https://api.osf.io/v2/nodes/{args.project}/files/osfstorage/{args.folder_id}/"
    output_dir = Path(args.output)

    print(f"Downloading from OSF project {args.project}...")
    download_folder(api_url, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
