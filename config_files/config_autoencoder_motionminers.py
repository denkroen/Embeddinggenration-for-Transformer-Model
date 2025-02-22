import os

DATASET_NAME = "motionminers"
EXPERIMENT_TYPE = "autoencoder"

# Training hyperparameters
VAL_BATCHES = 100
BATCH_SIZE = 100

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001


# Dataset
DATA_DIR = "/data/dkroen/dataset/motionminers/"
NUM_WORKERS = os.cpu_count()-2

WINDOW_LENGTH = 100
WINDOW_STEP = 12

NUM_SENSORS = 27
NUM_CLASSES = 7

NUM_FILTERS = 64
FILTER_SIZE = 5

# Compute related
ACCELERATOR = "gpu"
DEVICE = "2"
PRECISION = 32

# TCNN Related
MODE = "attribute" # / "classification"


#if attributes
NUM_ATTRIBUTES = 19
PATH_ATTRIBUTES = "atts_statistics_revised.txt"