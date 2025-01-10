import cv2
import numpy as np
from osgeo import gdal
from pathlib import Path

def preprocess_image(image_path, target_size=(224, 224)):
    # Open the image using GDAL to preserve geospatial information
    dataset = gdal.Open(str(image_path))
    
    # Read the image as a numpy array
    img = dataset.ReadAsArray()
    
    # Transpose the array to get the correct channel order (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    
    # Resize the image
    img_resized = cv2.resize(img, target_size)
    
    # Normalize the image
    img_normalized = img_resized / 255.0
    
    return img_normalized, dataset.GetGeoTransform()

def process_dataset(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for img_path in data_dir.glob('*.tif'):
        img, geotransform = preprocess_image(img_path)
        
        output_path = output_dir / f"{img_path.stem}_processed.npz"
        np.savez(output_path, image=img, geotransform=geotransform)

# Usage
process_dataset('path/to/raw/images', 'path/to/processed/images')