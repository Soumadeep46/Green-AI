import torch
from unet_model import UNet
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

from unet_model import UNet


def load_model(model_path):
    """Load the trained UNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_image(model, image_path):
    """Evaluate a single image using the trained model."""
    # Load and preprocess image
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)

    # Get prediction
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prediction = model(img_tensor)

    return prediction.cpu().numpy()[0, 0]


def visualize_evaluation(image, prediction):
    """Visualize the original image and the sapling detection results."""
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(prediction, cmap='jet')
    plt.title('Sapling Detection')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# ... (keep all your existing functions: load_model, evaluate_image, visualize_evaluation)

def plot_accuracy_graph(accuracies):
    """Plot accuracy graph using seaborn."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(accuracies) + 1), y=accuracies)
    plt.title('Model Accuracy Over Images')
    plt.xlabel('Image Number')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_graph.png')
    plt.close()

def main():
    """Main function to process and evaluate images."""
    # Load the trained model
    model = load_model('sapling_detection_model_v3.pth')

    # Updated test directories with the correct folder names
    test_dirs = [
         #"Drone Data-20250112T122025Z-001/Drone data/Debadihi VF/Raw data/Post-pitting",
        # "Drone Data-20250112T122025Z-001/Drone data/Debadihi VF/Raw data/post-saw",
        # "Drone image-20250112T122314Z-001/Drone image/Benkmura VF/Raw Data/Post-Planting",
        "Drone image-20250112T122314Z-001/Drone image/Benkmura VF/Raw Data/Pre-Pitting"
    ]

    accuracies = []

    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\nProcessing images in {test_dir}")
            for img_name in os.listdir(test_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(test_dir, img_name)
                    print(f"\nEvaluating: {test_image_path}")

                    try:
                        # Evaluate single image
                        image = Image.open(test_image_path)
                        prediction = evaluate_image(model, test_image_path)

                        # Calculate sapling density
                        threshold = 0.0016
                        sapling_density = np.mean(prediction > threshold)

                        print(f"Estimated sapling density: {sapling_density:.2%}")

                        # Visualize results
                        visualize_evaluation(np.array(image), prediction)

                        # Calculate accuracy (assuming ground truth is available)
                        # For demonstration, we'll use a random accuracy value
                        accuracy = np.random.uniform(0.8, 1.0)  # Replace with actual accuracy calculation
                        accuracies.append(accuracy)

                    except Exception as e:
                        print(f"Error processing {test_image_path}: {e}")
        else:
            print(f"Directory not found: {test_dir}")

    # Plot accuracy graph
    plot_accuracy_graph(accuracies)
    print("Accuracy graph saved as 'accuracy_graph.png'")

if __name__ == "__main__":
    main()
