import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

#import config_autoencoder_mocap as config
import config_files.config_transformer_gnn_mbientlab as config


from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from tcnn_module import TCNN_module
from LARA_data_module import LARADataModule

from tcnn_autoencoder import TCNN_Autoencoder_module

from embedding_transformer_module import EmbeddingTransformer
import torch
import torch.multiprocessing


torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    #seeding
    seed_everything(42, workers=True)


    torch.multiprocessing.set_sharing_strategy('file_system')

    # Shortcuts for Hyperparameters
    #config.MODE = "attribute" #shortcut
    #config.TIME_TOKENS = False
    #config.LEARNABLE = True
    #config.DATA_DIR = "/data/dkroen/dataset/mbientlab100/"
    #config.WINDOW_LENGTH = 100
    #config.LEARNING_RATE = 0.001
    #config.BATCH_SIZE = 50
    #config.EMBEDDING_SIZE = 64
    #config.NUM_LAYERS = 6
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3" #config.DEVICE





    DATA_FOLDER = "/data/dkroen/"
    MONITOR_LOSS = "validation_loss"


    #checkpoints
    CHECKPOINT_PATH = DATA_FOLDER + config.MODE + "/" + "pl_results/" + config.DATASET_NAME + "/" + config.EXPERIMENT_TYPE + "/"



    if config.EXPERIMENT_TYPE == "transformer":
        CHECKPOINT_PATH = CHECKPOINT_PATH + config.GENERATOR + "/"
        CHECKPOINT_PATH_LOAD = DATA_FOLDER + config.MODE + "/" + "pl_results/" + config.DATASET_NAME + "/" + config.GENERATOR + "/" + config.CHECKPOINT

        if config.PRETRAINED:
            CHECKPOINT_PATH = CHECKPOINT_PATH + "pretrained/"
        if config.FREEZE:
            CHECKPOINT_PATH = CHECKPOINT_PATH + "freeze/"



    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)


    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor=MONITOR_LOSS,
        mode="min",
        dirpath=CHECKPOINT_PATH,
        filename="checkpoint-Autoencoder-{epoch:02d}-{validation_loss:.7f}",
    )    

    TENSORBOARD_PATH = DATA_FOLDER + config.MODE + "/" + "logs/" + config.DATASET_NAME + "/" + config.EXPERIMENT_TYPE + "/"
    TENSORBOARD_FILE = config.EXPERIMENT_TYPE
    if config.EXPERIMENT_TYPE == "transformer":
        TENSORBOARD_PATH = TENSORBOARD_PATH + config.GENERATOR + "/"
        TENSORBOARD_FILE = TENSORBOARD_FILE + "_" + config.GENERATOR

        if config.PRETRAINED:
            TENSORBOARD_PATH = TENSORBOARD_PATH + "pretrained/"
        if config.FREEZE:
            TENSORBOARD_PATH = TENSORBOARD_PATH + "freeze/"


    logger = TensorBoardLogger(TENSORBOARD_PATH + "tb_logs", name= TENSORBOARD_FILE + "-logs")
    #profiler = PyTorchProfiler(
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler(DATA_FOLDER + "logs/" + config.DATASET_NAME + "/" + config.EXPERIMENT_TYPE + "/" + "tb_logs" "/" + "profiler0"),
    #    schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    #)

    '''model = TCNN(
        learning_rate=config.LEARNING_RATE,
        num_filters=config.NUM_FILTERS,
        filter_size=config.FILTER_SIZE,
        mode=config.MODE,
        num_attributes=config.NUM_ATTRIBUTES,
        num_classes=config.NUM_CLASSES,
        window_length=config.WINDOW_LENGTH,
        sensor_channels=config.NUM_SENSORS,
        path_attributes=config.PATH_ATTRIBUTES,
        
    )'''
    if config.EXPERIMENT_TYPE == "tcnn":

        model = TCNN_module(
            learning_rate=config.LEARNING_RATE,
            num_filters=config.NUM_FILTERS,
            filter_size=config.FILTER_SIZE,
            mode=config.MODE,
            num_attributes=config.NUM_ATTRIBUTES,
            num_classes=config.NUM_CLASSES,
            window_length=config.WINDOW_LENGTH,
            sensor_channels=config.NUM_SENSORS,
            path_attributes=config.PATH_ATTRIBUTES
        )

    elif config.EXPERIMENT_TYPE == "autoencoder":

        model = TCNN_Autoencoder_module(
            learning_rate=config.LEARNING_RATE,
            num_filters=config.NUM_FILTERS,
            filter_size=config.FILTER_SIZE,
            mode=config.MODE,
            num_attributes=config.NUM_ATTRIBUTES,
            num_classes=config.NUM_CLASSES,
            window_length=config.WINDOW_LENGTH,
            sensor_channels=config.NUM_SENSORS,
            path_attributes=config.PATH_ATTRIBUTES
        )
    
    elif config.EXPERIMENT_TYPE == "transformer":

        checkpoint_path = CHECKPOINT_PATH + config.CHECKPOINT

        model = EmbeddingTransformer(
            learning_rate=config.LEARNING_RATE,
            num_filters=config.NUM_FILTERS,
            filter_size=config.FILTER_SIZE,
            mode=config.MODE,
            num_attributes=config.NUM_ATTRIBUTES,
            num_classes=config.NUM_CLASSES,
            window_length=config.WINDOW_LENGTH,
            sensor_channels=config.NUM_SENSORS,
            path_attributes=config.PATH_ATTRIBUTES,
            n_trans_layers=config.TRANS_LAYERS,
            n_trans_hidden_neurons=config.TRANS_HIDDEN_NEURONS,
            embedding_model=config.GENERATOR,
            pretrained=config.PRETRAINED,
            freeze=config.FREEZE,
            time_tokens=config.TIME_TOKENS,
            num_layers=config.NUM_LAYERS,
            embedding_size=config.EMBEDDING_SIZE,
            checkpoint_path=CHECKPOINT_PATH_LOAD,
            learnable_adj=config.LEARNABLE,
            GSL = config.GSL
            
        )

    data_module = LARADataModule(
        datadir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        #accelerator=config.ACCELERATOR,
    )
    
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices= [0],
        #profiler=profiler,
        logger=logger,
        min_epochs=5,
        max_epochs=config.NUM_EPOCHS + 100,
        precision=config.PRECISION,
        # limit_val_batches=1000,
        accumulate_grad_batches=4,
        val_check_interval=config.VAL_BATCHES,
        callbacks=[EarlyStopping(monitor=MONITOR_LOSS,patience=30),checkpoint_callback],
        deterministic=True
    )


    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)
