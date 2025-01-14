# Green AI

This project uses machine learning and image processing techniques to monitor afforestation efforts using drone imagery. It analyzes drone images to detect saplings, estimate their survival rate, and identify the locations of dead saplings.

# Features

- Preprocess drone images while preserving geospatial information
- Train a U-Net model for sapling detection and survival prediction
- Analyze new drone images to estimate sapling survival rates
- Visualize results with side-by-side comparisons of original and predicted images
- Calculate and display geo-coordinates of identified dead saplings
- Parallel processing capability for analyzing multiple images efficiently

# Files in the Repository

1. `preprocess_images.py`: Preprocesses raw drone images and saves them in a format suitable for model training.
2. `unet_model.py`: Defines the U-Net architecture used for sapling detection and survival prediction.
3. `train_model.py`: Loads preprocessed data, trains the U-Net model, and saves the trained model.
4. `analyze_images.py`: Uses the trained model to analyze new drone images, visualize results, and calculate survival rates.
5. `optimize_processing.py`: Implements parallel processing for efficient analysis of multiple images.
6. `eval.py`: To evaluate the trained model.

## Installation Instructions
### Prerequisites
To run this project, you will need Python and the following libraries:

NumPy: For data handling and numerical computations.

PyTorch: For model loading, inference, and tensor operations.

Scikit-learn: For calculating evaluation metrics (e.g., precision, recall, F1-score).

Matplotlib: For visualizing images and predictions.

Torchvision: For image processing utilities (optional, not directly used in the code).

Pillow: For loading and preprocessing images.

You can install the necessary dependencies using pip:

```bash
  pip install numpy torch scikit-learn matplotlib torchvision pillow  scipy opencv-python-headless

```

### GPU Support (Optional)
For improved performance, especially when training the model, it's recommended to use a machine with a GPU. Ensure that TensorFlow is configured to use the GPU.


```bash
  pip install tensorflow-gpu

```

