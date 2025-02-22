import os

DATASET_NAME = "motionsense"
EXPERIMENT_TYPE = "transformer"
GENERATOR = "baseline"

# Generator Hyperparameters
FREEZE = False
PRETRAINED = False
CHECKPOINT = "checkpoint-Autoencoder-epoch=00-validation_loss=0.34.ckpt"

# For MLP Generator
NUM_LAYERS = 3 #number of MLP-Layers

# For MLP Generator and baseline
EMBEDDING_SIZE = 64 #expansion in sensor dimention

#Transformer Hyperparameters
TRANS_LAYERS = 6
TRANS_HIDDEN_NEURONS = 128

#Use time dimention as token
TIME_TOKENS = False

# Training hyperparameters
VAL_BATCHES = 100
BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# Dataset
DATA_DIR = "/data/dkroen/dataset/motionsense/"
NUM_WORKERS = os.cpu_count()-2

WINDOW_LENGTH = 50
WINDOW_STEP = 0

NUM_SENSORS = 9
NUM_CLASSES = 6

#tcnn
NUM_FILTERS = 64
FILTER_SIZE = 5

# Compute related
ACCELERATOR = "gpu"
DEVICE = "0"
PRECISION = 32

# TCNN Related
MODE = "classification"
#if attributes
NUM_ATTRIBUTES = 19
PATH_ATTRIBUTES = "atts_statistics_revised.txt"
