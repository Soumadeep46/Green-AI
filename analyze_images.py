import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from osgeo import gdal
from preprocess_images import preprocess_image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def analyze_image(model, image_path):
    img, geotransform = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction[0, :, :, 0], geotransform

def visualize_results(image_path, prediction, geotransform):
    # Open the original image
    dataset = gdal.Open(image_path)
    img = dataset.ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot the original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot the prediction
    ax2.imshow(prediction, cmap='jet', alpha=0.7)
    ax2.set_title('Sapling Survival Prediction')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_survival_rate(prediction, threshold=0.5):
    total_saplings = np.sum(prediction > threshold)
    total_area = prediction.shape[0] * prediction.shape[1]
    return total_saplings / total_area

# Load the trained model
model = load_model('sapling_survival_model.h5')

# Analyze a new image
image_path = 'path/to/new/drone/image.tif'
prediction, geotransform = analyze_image(model, image_path)

# Visualize the results
visualize_results(image_path, prediction, geotransform)

# Calculate survival rate
survival_rate = calculate_survival_rate(prediction)
print(f"Estimated sapling survival rate: {survival_rate:.2%}")

# Identify locations of dead saplings
dead_saplings = np.where(prediction < 0.5)
print(f"Number of identified dead saplings: {len(dead_saplings[0])}")

# Convert pixel coordinates to geo-coordinates
pixel_to_geo = lambda x, y: (geotransform[0] + x*geotransform[1] + y*geotransform[2],
                             geotransform[3] + x*geotransform[4] + y*geotransform[5])

dead_sapling_coords = [pixel_to_geo(x, y) for x, y in zip(dead_saplings[1], dead_saplings[0])]
print("Geo-coordinates of dead saplings:")
for coord in dead_sapling_coords[:10]:  # Print first 10 coordinates
    print(f"Longitude: {coord[0]}, Latitude: {coord[1]}")