import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dogbreed_datamodule import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper

@task_wrapper
def train_and_test(data_module, model, trainer):
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)

def main():
    # Set up paths
    base_dir = Path('..')
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    
    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    data_module = DogBreedImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)

    # Initialize Model
    model = DogBreedClassifier(lr=1e-3)

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir / "dogbreed_classification" / "checkpoints",
        filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=5,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name="dogbreed_classification",
        ),
    )

    # Train and test the model
    train_and_test(data_module, model, trainer)

if __name__ == "__main__":
    main()