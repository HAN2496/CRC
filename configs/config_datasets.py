# Data loader configuration

TD_SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
CP_SUBJECTS = [2, 3]

TD_FOLDER = "EPIC"
CP_FOLDER = "CP child gait data"

INPUT_WINDOW_LENGTH = 400
STRIDE = 50

# --------------------------------------------
# Model configuration

MODEL_NAME = 'InceptionTime'


# --------------------------------------------
# Training configuration

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
