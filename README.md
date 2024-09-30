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
<img width="751" alt="image" src="https://github.com/user-attachments/assets/63ceb754-987c-4b9d-aaed-ff78a9ccef60">


We can view the files created inside the docker container using the command:
```bash 
docker-compose run train /bin/bash
```
The following is the output:

<img width="521" alt="image" src="https://github.com/user-attachments/assets/53123c64-a746-471d-82c3-2f049d1f3485">


### Evaluation

To perform evaluation of the model, use the `eval` service:
```bash
docker-compose up eval
```
This will evaluate the model on the validation dataset and output the results.
The following is the output of the command on gitpod:

<img width="752" alt="image" src="https://github.com/user-attachments/assets/1f593925-3d58-4b01-9d4b-166fdf7b4bed">


We can view the files created inside the docker container using the command:
```bash 
docker-compose run eval /bin/bash
```
The following is the output:

<img width="575" alt="image" src="https://github.com/user-attachments/assets/db7b13ce-20e7-48c0-90fa-20c7e786698d">



### Inference

To perform inference on new images, use the `infer` service:
```bash
docker-compose up infer
```
You can place your images in the `output/` directory to see the results after the inference completes.

The following is the output of the command on gitpod:

<img width="755" alt="image" src="https://github.com/user-attachments/assets/1a2d532c-deaf-49cd-9c01-97f378f7b0d6">


We can view the files created inside the docker container using the command:
```bash 
docker-compose run infer /bin/bash
```
The following is the output:

<img width="569" alt="image" src="https://github.com/user-attachments/assets/b948f5a8-5234-4b98-9d98-b9d94f8419aa">

The following are the images in the `output/` directory

<table border="0">
<tr>
  <td><img src="https://github.com/user-attachments/assets/defb814f-1f46-47eb-bb5b-0ea8ec62925d" width="500" /></td>
  <td><img src="https://github.com/user-attachments/assets/c6b06c26-ddbb-4730-81c9-59e33ef91540" width="500" /></td>
  <td><img src="https://github.com/user-attachments/assets/e58ae322-21bf-4455-9247-2d739b50ac30" width="500" /></td>
  <td><img src="https://github.com/user-attachments/assets/d5db4274-4ef1-44bd-a82a-30be1f0c54b8" width="500" /></td>
</tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/234dec11-caa3-4501-a814-4bc9297a0486" width="500" /></td>
   <td><img src="https://github.com/user-attachments/assets/10482bdb-4ab1-480b-ad37-b40f5a1b7ca9" width="500" /></td>
    <td><img src="https://github.com/user-attachments/assets/4db979dd-0e1d-40ce-946e-330b08d79c1e" width="500" /></td>
    <td><img src="https://github.com/user-attachments/assets/bacb00a1-f73b-4a19-b79e-e14eff29d143" width="500" /></td>
  </tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/9035a418-c20d-4d38-a31e-d4aceef352a5" width="500" /></td>
  <td><img src="https://github.com/user-attachments/assets/a3a7d602-6dcb-4891-9f8f-2ee2cbc97115" width="500" /></td>

</tr>
</table>

## Results

After training, the model's performance metrics and checkpoints will be available in the `logs/` directory.

## Volume Mounts

The Docker Compose file defines several named volumes to manage data and logs:

- **data**: Stores the training and inference data.
- **logs**: Centralized logging for training and inference.
- **output**: Stores the output of the inference process.


