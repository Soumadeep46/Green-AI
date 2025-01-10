import multiprocessing
import tensorflow as tf
from analyze_images import analyze_image, calculate_survival_rate

def process_image(args):
    model, image_path = args
    prediction, geotransform = analyze_image(model, image_path)
    survival_rate = calculate_survival_rate(prediction)
    return image_path, survival_rate

def parallel_processing(model, image_paths):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    
    results = pool.map(process_image, [(model, path) for path in image_paths])
    
    pool.close()
    pool.join()
    
    return dict(results)

# Enable GPU acceleration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the model
model = tf.keras.models.load_model('sapling_survival_model.h5')

# List of image paths to process
image_paths = ['path/to/image1.tif', 'path/to/image2.tif', 'path/to/image3.tif']

# Process images in parallel
results = parallel_processing(model, image_paths)

# Print results
for path, survival_rate in results.items():
    print(f"Image: {path}, Survival Rate: {survival_rate:.2%}")