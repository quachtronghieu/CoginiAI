import os
import sys
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from build_default_assets import (
    ensure_dir, copy_file, process_emoji_collection,
    generate_index_json, generate_config_json, pack_assets_simple
)

def build():
    emoji_dir = str(Path(__file__).resolve().parent / 'emojis_240')
    output_dir = str(Path(__file__).resolve().parent / 'output')
    assets_bin_target = str(BASE_DIR / 'assets.bin')

    ensure_dir(output_dir)
    temp_build_dir = os.path.join(output_dir, 'temp_build')
    assets_dir = os.path.join(temp_build_dir, 'assets')
    if os.path.exists(temp_build_dir):
        shutil.rmtree(temp_build_dir)
    ensure_dir(temp_build_dir)
    ensure_dir(assets_dir)

    emoji_list = process_emoji_collection(emoji_dir, assets_dir)
    print(f"Total emojis processed: {len(emoji_list)}")
    for e in emoji_list:
        print(f" - {e['name']} -> {e['file']}")

    generate_index_json(assets_dir, None, None, emoji_list, None, None)
    config_path = generate_config_json(temp_build_dir, assets_dir)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    include_path = config_data['include_path']
    image_file = config_data['image_file']
    pack_assets_simple(assets_dir, include_path, image_file, 'assets', int(config_data['name_length']))

    if os.path.exists(image_file):
        shutil.copy2(image_file, assets_bin_target)
        size_kb = os.path.getsize(assets_bin_target) / 1024
        print(f"SUCCESS: Generated {assets_bin_target} ({size_kb:.2f} KB)")
    else:
        print("ERROR: Failed to generate assets.bin")

if __name__ == '__main__':
    build()
