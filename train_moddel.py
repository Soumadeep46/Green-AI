import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from unet_model import unet_model
from pathlib import Path


def load_data(data_dir):
    X = []
    y = []
    for npz_file in Path(data_dir).glob('*_processed.npz'):
        data = np.load(npz_file)
        X.append(data['image'])
        # Assume we have corresponding mask files
        mask = np.load(npz_file.with_name(f"{npz_file.stem}_mask.npy"))
        y.append(mask)
    return np.array(X), np.array(y)

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = unet_model()
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=16)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")
    
    return model, history

# Load the data
X, y = load_data('path/to/processed/images')

# Train the model
model, history = train_model(X, y)

# Save the model
model.save('sapling_survival_model.h5')