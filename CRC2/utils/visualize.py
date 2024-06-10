import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('results'):
    os.makedirs('results')

def visualize_predictions(model, X_test, y_test, n_samples=6, input_window_length=400, stride=1):
    """
    Visualizes predicted and actual angles for sliding windows within given test sequences.
    
    Args:
    - model (tsai Learner): The trained model.
    - X_test (list of np.array): List of test sequences.
    - y_test (list of np.array): List of actual target sequences (x, y).
    - n_samples (int): Number of random samples to display.
    - input_window_length (int): The length of each input window.
    - stride (int): The stride of the sliding window.
    """

    # Plot samples
    n_cols = 2
    n_rows = (n_samples + 1) // n_cols
    plt.figure(figsize=(15, 3 * n_rows))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        sequence = X_test[idx]
        target_sequence = y_test[idx]
        preds = []
        actuals = []

        # Calculate the middle point of the sequence
        mid_point = len(sequence) // 2
        start_point = max(0, mid_point - 900)  # Start 500 steps before the middle
        end_point = min(len(sequence), mid_point + 500)  # End 500 steps after the middle

        print(f"Sequence {idx + 1} - Length: {len(sequence)} from step {start_point} to {end_point}")

        # Iterate over the middle 1000 steps with a sliding window
        for start in range(start_point, end_point - input_window_length, stride):
            end = start + input_window_length
            window = sequence[start:end].reshape(1, input_window_length, -1)
            
            # Predict using the trained model
            _, _, pred = model.get_X_preds(window)  # pred is expected to be [cos, sin] for each window
            pred = np.array(pred)

            # Convert predictions from [cos, sin] to gait phase percentage
            pred_percentage = np.arctan2(pred[0, 1], pred[0, 0]) * 100 / (2 * np.pi) + 50
            actual_percentage = np.arctan2(target_sequence[end-1][1], target_sequence[end-1][0]) * 100 / (2 * np.pi) + 50
            
            preds.append(pred_percentage)
            actuals.append(actual_percentage)

        # Plot predictions and actual values
        ax1 = plt.subplot(n_rows, n_cols, i + 1)
        ax1.plot(actuals, label='Actual Gait Phase', color='green', alpha=0.5)
        ax1.plot(preds, label='Predicted Gait Phase', color='red')
        # ax1.set_ylabel('Gait Phase (%)')

        # Add second y-axis for hip_sagittal data
        ax2 = ax1.twinx()
        ax2.plot(sequence[(start_point+input_window_length):end_point, 0], label='Input Hip Sagittal', linestyle='--', alpha=0.5)
        # ax2.set_ylabel('Hip Sagittal Angle (degrees)')
        ax2.tick_params(axis='y')

        # plt.title(f'Sequence {idx + 1}')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()

    plt.savefig('results/estimation_visualization.png', format='png', dpi=300)
    plt.show()

def visualize_gp(original_datasets, gp):
    plt.figure(figsize=(12, 8))
    plt.plot(original_datasets['header'], original_datasets['hip_sagittal'], label='Actual Data (y)')
    plt.plot(original_datasets['header'], gp.y_pred, label='Predicted Data (y_pred)')
    plt.fill_between(original_datasets['header'], gp.y_pred - gp.sigma, gp.y_pred + gp.sigma,
                    color='blue', alpha=0.2, label='Confidence Interval (1 std dev)')
    plt.legend('Reference trajectory')
    plt.savefig('results/reference trajectory visualization.png', format='png', dpi=300)
    plt.show()