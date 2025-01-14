import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import warnings

# Increase PIL's image size limit (use with caution)
Image.MAX_IMAGE_PIXELS = None


def process_image(img_path):
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((256, 256))
            return np.array(img) / 255.0
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def process_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    images = []
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_name)
            img_array = process_image(img_path)
            if img_array is not None:
                images.append(img_array)
    return np.stack(images) if images else None


def process_tif(tif_path, tile_size=1024):
    if not os.path.exists(tif_path):
        print(f"TIF file not found: {tif_path}")
        return None

    try:
        with Image.open(tif_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            tiles = []

            for i in range(0, width, tile_size):
                for j in range(0, height, tile_size):
                    box = (i, j, min(i + tile_size, width), min(j + tile_size, height))
                    tile = img.crop(box)
                    tile = tile.resize((256, 256))
                    tiles.append(np.array(tile) / 255.0)

            return np.stack(tiles)
    except Exception as e:
        print(f"Error processing TIF {tif_path}: {e}")
        return None


def preprocess_data(base_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Process Benkmura VF images
    benkmura_ortho = process_tif(os.path.join(base_path, "Benkmura VF", "Ortho", "Post Pitting", "Pitting.tif"))
    benkmura_post = process_folder(os.path.join(base_path, "Benkmura VF", "Raw Data", "Post-Planting"))
    benkmura_pre = process_folder(os.path.join(base_path, "Benkmura VF", "Raw Data", "Pre-Pitting"))

    # Process Debadihi VF images using absolute paths
    debadihi_post = process_folder(r"E:\GREEN AI\Drone Data-20250112T122025Z-001\Drone Data\Debadihi VF\Raw Data\Post-Pitting")
    debadihi_sw = process_folder(r"E:\GREEN AI\Drone Data-20250112T122025Z-001\Drone Data\Debadihi VF\Raw Data\post-SW")

    # Save processed data
    if benkmura_ortho is not None:
        np.save(os.path.join(output_path, "benkmura_ortho.npy"), benkmura_ortho)
    if benkmura_post is not None:
        np.save(os.path.join(output_path, "benkmura_post.npy"), benkmura_post)
    if benkmura_pre is not None:
        np.save(os.path.join(output_path, "benkmura_pre.npy"), benkmura_pre)
    if debadihi_post is not None:
        np.save(os.path.join(output_path, "debadihi_post.npy"), debadihi_post)
    if debadihi_sw is not None:
        np.save(os.path.join(output_path, "debadihi_sw.npy"), debadihi_sw)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    base_dir = r"E:\GREEN AI\Drone image-20250112T122314Z-001\Drone image"
    output_dir = "processed_data"
    preprocess_data(base_dir, output_dir)
