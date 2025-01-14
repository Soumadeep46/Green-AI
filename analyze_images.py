import torch
import numpy as np
from PIL import Image
from unet_model import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def analyze_image(model, image_path, threshold=0.7):  # Increased threshold
    # Load and preprocess image
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_size = img.size
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0

        # Normalize the input
        img_array = (img_array - img_array.mean()) / img_array.std()
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)

    # Get prediction
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prediction = model(img_tensor)

    return prediction.cpu().numpy()[0, 0], original_size


def calculate_metrics(prediction, threshold=0.7):
    binary_pred = prediction > threshold

    # Calculate density
    sapling_density = np.mean(binary_pred)

    # Calculate connected components (rough estimate of sapling count)
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(binary_pred)

    return {
        'density': sapling_density,
        'estimated_count': num_features,
        'confidence_mean': np.mean(prediction),
        'confidence_std': np.std(prediction)
    }


def visualize_results(image_path, prediction, metrics, threshold=0.7):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(131)
    plt.imshow(img_array)
    plt.title("Original Image")

    # Prediction heatmap
    plt.subplot(132)
    plt.imshow(prediction, cmap='jet')
    plt.title("Sapling Detection Heatmap")
    plt.colorbar()

    # Binary prediction
    plt.subplot(133)
    plt.imshow(prediction > threshold, cmap='binary')
    plt.title(f"Binary Prediction (threshold={threshold})")

    # Add metrics text
    plt.figtext(0.02, 0.02,
                f"Density: {metrics['density']:.2%}\n"
                f"Est. Count: {metrics['estimated_count']}\n"
                f"Conf Mean: {metrics['confidence_mean']:.3f}\n"
                f"Conf Std: {metrics['confidence_std']:.3f}",
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main(model_path, image_path, threshold=0.7):
    model = load_model(model_path)
    prediction, original_size = analyze_image(model, image_path, threshold)
    metrics = calculate_metrics(prediction, threshold)

    print("\nAnalysis Results:")
    print(f"Sapling Density: {metrics['density']:.2%}")
    print(f"Estimated Sapling Count: {metrics['estimated_count']}")
    print(f"Confidence Mean: {metrics['confidence_mean']:.3f}")
    print(f"Confidence Std: {metrics['confidence_std']:.3f}")

    visualize_results(image_path, prediction, metrics, threshold)


if __name__ == "__main__":
    main("sapling_detection_model_v2.pth", "Drone image-20250112T122314Z-001/Drone image/Benkmura VF/Raw Data/Pre-Pitting", threshold=0.7)
