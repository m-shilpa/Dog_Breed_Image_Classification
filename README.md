# Dog Breed Classification

This repository contains a project for training and performing inference on Kaggle's Dog Breed Images Dataset using Docker Compose. The project is divided into three main services: `train`, `eval`, and `infer`.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Volume Mounts](#volume-mounts)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project aims to classify dog breeds from images using a convolutional neural network (CNN) model. The training and inference processes are containerized using Docker Compose for ease of deployment and reproducibility. This project can be useful for applications in pet adoption, veterinary services, and more.

## Dataset

### Dog Breed Image Dataset

This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks. Each breed is represented by 100 images, stored in separate directories named after the respective breed. The dataset includes the following breeds:

1. Beagle
2. Boxer
3. Bulldog
4. Dachshund
5. German Shepherd
6. Golden Retriever
7. Labrador Retriever
8. Poodle
9. Rottweiler
10. Yorkshire Terrier

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset).

### Test Images

Additionally, a set of 10 test images (`dog_breed_10_test_images.zip`) is included for testing purposes. These images are used to evaluate the model's performance on unseen data.

## Project Structure

```
.
├── src
│   ├── datamodules
│   │   └── dogbreed_datamodule.py
│   ├── models
│   │   └── dogbreed_classifier.py
│   ├── utils
│   │   └── logging_utils.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── dog_breed_10_test_images.zip
├── dog_breed_image_dataset.zip
├── requirements.txt
```

## Setup

### Prerequisites

- Ensure you have Docker and Docker Compose installed on your machine:
  - [Docker Installation Guide](https://docs.docker.com/get-docker/)
  - [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/m-shilpa/Dog_Breed_Image_Classification
   cd dog-breed-classification
   ```

2. **Build and start the Docker containers:**
   ```bash
   docker compose build --no-cache
   docker compose up
   ```

## Usage

### Training

To train the model, run:
```bash
docker-compose run train
```
This command will initiate the training process using the training dataset. The model's performance metrics and checkpoints will be available in the `logs/` directory.

### Evaluation

To perform evaluation of the model, use the `eval` service:
```bash
docker-compose run eval
```
This will evaluate the model on the validation dataset and output the results.

### Inference

To perform inference on new images, use the `infer` service:
```bash
docker-compose run infer
```
You can place your images in the `output/` directory to see the results after the inference completes.

## Results

After training, the model's performance metrics and checkpoints will be available in the `logs/` directory.

## Volume Mounts

The Docker Compose file defines several named volumes to manage data and logs:

- **data**: Stores the training and inference data.
- **logs**: Centralized logging for training and inference.
- **output**: Stores the output of the inference process.

## Troubleshooting

If you encounter issues during setup, consider checking:
- Your Docker installation and permissions.
- Compatibility of your Docker Compose file version.
- Any errors in the console output for specific troubleshooting steps.
