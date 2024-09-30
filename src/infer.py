import argparse
from pathlib import Path
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever'
                    , 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@task_wrapper
def main(args):
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()

    if args.input_folder == None:
        """Download images and prepare images datasets."""
        download_and_extract_archive(
            url="https://github.com/m-shilpa/lightning-template-hydra/raw/main/dog_breed_10_test_images.zip",
            download_root='./',
            remove_finished=True
        )
        input_folder = base_dir / 'dog_breed_10_test_images'
    else:
        input_folder = Path(args.input_folder)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    image_files = list(input_folder.glob('*'))
    print(image_files)
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))
        
        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)
                
                progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

if __name__ == "__main__":

    ckpt_file_path = f'/workspace/logs/dogbreed_classification/checkpoints/{os.listdir("/workspace/logs/dogbreed_classification/checkpoints/")[-1]}'

    parser = argparse.ArgumentParser(description="Infer using trained DogBreed Classifier")
    parser.add_argument("--input_folder", type=str, required=False, default=None, help="Path to input folder containing images")
    parser.add_argument("--output_folder", type=str, required=False, default='../output', help="Path to output folder for predictions")
    parser.add_argument("--ckpt_path", type=str, required=False,default=ckpt_file_path, help="Path to model checkpoint")
    args = parser.parse_args()

    print(args)
    
    base_dir = Path('/workspace')
    log_dir = base_dir / "logs"
    setup_logger(log_dir / "infer_log.log")

    main(args)
