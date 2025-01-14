import torch
import torch.multiprocessing as mp
from analyze_images import analyze_image, calculate_survival_rate, load_model

def process_image(args):
    model, image_path = args
    prediction = analyze_image(model, image_path)
    survival_rate = calculate_survival_rate(prediction)
    return image_path, survival_rate

def parallel_processing(model_path, image_paths):
    # Load model
    model = load_model(model_path)
    
    # Use all available CPU cores
    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_image, [(model, path) for path in image_paths])
    
    return dict(results)

if __name__ == "__main__":
    model_path = "sapling_detection_model.pth"
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
    
    results = parallel_processing(model_path, image_paths)
    
    for path, survival_rate in results.items():
        print(f"Image: {path}, Survival Rate: {survival_rate:.2%}")
