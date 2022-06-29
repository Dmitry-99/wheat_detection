from trainer.trainer import *
from trainer.plots import plot_loss, plot_img
from trainer.config import CFG
from trainer.utils import format_prediction_string

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    trainer = Trainer()
    if CFG.mode == 'train':
        total_train_loss, total_val_loss = trainer.run('train')
        plot_loss(total_train_loss, total_val_loss, save_path='experiments/plots/loss.png')
    if CFG.mode == 'inference':
        trainer.run('inference', CFG.inference_image_path)
        