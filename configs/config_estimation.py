# Data loader configuration

SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
# SUBJECTS = [6, 8, 9, 10, 11, 12]

INPUT_WINDOW_LENGTH = 400
STRIDE = 50

# --------------------------------------------
# Model configuration
"""
Arch로 가능한 것
 - InceptionTimePlus: 29        0.023866    0.036537    0.137970  00:02
 - InceptionTime: 
 - LSTMPlus: 29        0.011171    0.033059    0.129901  00:01
 - GRUPlus: 29        0.011756    0.032508    0.129282  00:00
 - RNNPlus: 29        0.013765    0.036125    0.139520  00:01 
확인해본 것
 - LSTM / RNN

"""
ARCH = "GRUPlus"

# --------------------------------------------
# Training configuration

NUM_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

MODEL_NAME = f'{ARCH}_{NUM_EPOCHS}epochs'