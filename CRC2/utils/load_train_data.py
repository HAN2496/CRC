import numpy as np
from sklearn.model_selection import train_test_split
from utils.td import TD
from utils.cp import CP
from configs.config_datasets import TD_SUBJECTS, CP_SUBJECTS
import configs.config_estimation as config

input_window_length = config.INPUT_WINDOW_LENGTH
stride = config.STRIDE

def load_train_data(use_td=True):
    all_sequences, all_targets = [], []
    subjects = TD_SUBJECTS if use_td else CP_SUBJECTS
    for number in subjects:
        subject = TD(number=number, choose_one_dataset=False, extract_walking=True) if use_td else CP(number=number)
        each_data_len = subject.datas['length of each file']
        hip_sagittal = np.array(subject.datas['hip_sagittal'])
        heelstrike_x = np.array(subject.datas['heelstrike_x'])
        heelstrike_y = np.array(subject.datas['heelstrike_y'])
        idx = 0
        for data_len in each_data_len:
            idxs = list(range(idx, idx + data_len))
            sequences = hip_sagittal[idxs].reshape(-1, 1)
            targets = np.column_stack((heelstrike_x[idxs], heelstrike_y[idxs]))
            all_sequences.append(sequences)
            all_targets.append(targets)
            idx += data_len

    # Split the sequences into training and testing
    train_data, test_data, train_targets, test_targets = train_test_split(all_sequences, all_targets, test_size=0.2, shuffle=False)

    # Apply sliding window to the training data
    X_train, y_train = [], []
    for data, target in zip(train_data, train_targets):
        for start in range(0, len(data) - input_window_length, stride):
            end = start + input_window_length
            X_train.append(data[start:end])
            y_train.append(target[end] if end < len(target) else target[-1])  # Handle edge case for targets

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Number of test sequences: {len(test_data)}, Number of test target sequences: {len(test_targets)}")
    return X_train, y_train, test_data, test_targets

if __name__ == "__main__":
    load_train_data(use_td=True)