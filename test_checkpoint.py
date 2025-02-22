import pytorch_lightning as pl

#import config_autoencoder_mocap as config
import config_files.config as config



from tcnn_module import TCNN_module
from LARA_data_module import LARADataModule
from tcnn_autoencoder import TCNN_Autoencoder_module
from embedding_transformer_module import EmbeddingTransformer

import torch
torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == '__main__':

    #model = TCNN_module.load_from_checkpoint("")
    #model = TCNN_Autoencoder_module.load_from_checkpoint("")
    model = EmbeddingTransformer.load_from_checkpoint(config.CHECKPOINT)

    data_module = LARADataModule(
        datadir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices= [0],
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS + 80,
        precision=config.PRECISION,
        val_check_interval=config.VAL_BATCHES,
        deterministic=True
    )

    trainer.test(model, data_module)
