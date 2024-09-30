import os
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary

from datamodules.dogbreed_datamodule import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = list(checkpoint_dir.glob("epoch=*-val_loss=*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

@task_wrapper
def evaluate_model(data_module, model, trainer):
    trainer.test(model, data_module)

def main():
    # Set up paths
    base_dir = Path('/workspace')
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    checkpoint_dir = log_dir / "dogbreed_classification" / "checkpoints"

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    data_module = DogBreedImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    print(latest_checkpoint)

    # Load model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(latest_checkpoint)

    trainer = L.Trainer(
        logger=TensorBoardLogger(save_dir=log_dir, name="dogbreed_classification_eval"),
        callbacks=[RichProgressBar()]
    )

    evaluate_model(data_module, model, trainer)

if __name__ == "__main__":
    main()
