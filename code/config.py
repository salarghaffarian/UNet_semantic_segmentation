# config.py

class Config:
    # Dataset
    ROOT_DIR = 'data/PascalVOC_Dataset'
    IMAGE_SET_TRAIN = 'train'
    IMAGE_SET_VAL = 'val'

    # Model
    N_CLASS = 20            # Number of classes in your dataset
    IMAGE_CHANNELS = 3      # Number of channels in input images
    USE_BN_DROPOUT = False  # Use batch normalization and dropout layers in the model architecture
    DROPOUT_P = 0.5         # Probability of an element to be zeroed if dropout is used

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # Transformations
    IMAGE_SIZE = (256, 256)  # Resize images to this size

    # Paths
    CHECKPOINT_PATH = '../checkpoints/model_checkpoint.pth'