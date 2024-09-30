# Dog Breed Classification

This repository contains a project for training and performing inference on Kaggle's Dog Breed Images Dataset using Docker Compose. The project is divided into three main services: `train`, `eval`, and `infer`.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Development Container](#development-container)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Volume Mounts](#volume-mounts)

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
├── .devcontainer
│   └── devcontainer.json
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dog_breed_image_dataset.zip
└── dog_breed_10_test_images.zip
```

## Setup

### Prerequisites

- Ensure you have Docker and Docker Compose installed on your machine:
  - [Docker Installation Guide](https://docs.docker.com/get-docker/)
  - [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

### Development Container

This project includes a development container setup, allowing you to work in a consistent environment with all the necessary dependencies pre-installed. 

To use the development container:

1. **Open the Project in VSCode**: Ensure you have the Remote - Containers extension installed.
2. **Reopen in Container**: Use the command palette (Ctrl+Shift+P) and select **Remote-Containers: Reopen in Container**. This will build and open your development container.
3. **Access the Terminal**: Once the container is running, open a terminal in VSCode. You can now run commands like `python train.py` directly within the container environment.


### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/m-shilpa/Dog_Breed_Image_Classification
   cd dog-breed-classification
   ```

2. **Build and start the Docker containers:**
   ```bash
   docker compose build --no-cache
   ```
   The following is the output of the command on gitpod:
   ![image](https://github.com/user-attachments/assets/b8886aee-2074-4106-95f2-29951990a21c)


## Usage

### Training

To train the model, run:
```bash
docker-compose up train
```
This command will initiate the training process using the training dataset. The model's performance metrics and checkpoints will be available in the `logs/` directory.

The following is the output of the command on gitpod:
![image](https://github.com/user-attachments/assets/4b926b33-d5b6-4c3a-afeb-e9b448e8659c)


### Evaluation

To perform evaluation of the model, use the `eval` service:
```bash
docker-compose up eval
```
This will evaluate the model on the validation dataset and output the results.
The following is the output of the command on gitpod:
![image](https://github.com/user-attachments/assets/4623e8a4-a678-49ab-9a7c-e7736a51095f)


### Inference

To perform inference on new images, use the `infer` service:
```bash
docker-compose up infer
```
You can place your images in the `output/` directory to see the results after the inference completes.

The following is the output of the command on gitpod:
![image](https://github.com/user-attachments/assets/93ecdb2f-2f86-4632-9287-1ce43190f97a)

## Results

After training, the model's performance metrics and checkpoints will be available in the `logs/` directory.

## Volume Mounts

The Docker Compose file defines several named volumes to manage data and logs:

- **data**: Stores the training and inference data.
- **logs**: Centralized logging for training and inference.
- **output**: Stores the output of the inference process.


