import os

DATASET_NAME = "mocap"
EXPERIMENT_TYPE = "transformer"
GENERATOR = "autoencoder"

# Generator Hyperparameters
FREEZE = False
PRETRAINED = True
CHECKPOINT = "checkpoint-Autoencoder-epoch=03-validation_loss=0.000022744.ckpt"

# For MLP Generator
NUM_LAYERS = 3 #number of MLP-Layers

# For MLP Generator and baseline
EMBEDDING_SIZE = 128 #expansion in sensor dimention


#Transformer Hyperparameters
TRANS_LAYERS = 6
TRANS_HIDDEN_NEURONS = 128

#Use time dimention as token
TIME_TOKENS = True


# Training hyperparameters
VAL_BATCHES = 100
BATCH_SIZE = 40
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# Dataset
DATA_DIR = "/data/dkroen/dataset/mocap/"
NUM_WORKERS = os.cpu_count()-2

WINDOW_LENGTH = 100
WINDOW_STEP = 12

NUM_SENSORS = 126
NUM_CLASSES = 7

NUM_FILTERS = 64
FILTER_SIZE = 5

# Compute related
ACCELERATOR = "gpu"
DEVICE = "1"
PRECISION = 32

# TCNN Related
MODE = "attribute" # / "classification"
#if attributes
NUM_ATTRIBUTES = 19
PATH_ATTRIBUTES = "atts_statistics_revised.txt"
