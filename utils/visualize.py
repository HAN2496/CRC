import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random
from tqdm import tqdm
import imageio
import glob

if not os.path.exists('results'):
    os.makedirs('results')

def visualize_predictions_scalar(control_target, model, X_test, y_test, n_samples=6, input_window_length=400, stride=1, file_name=None, show=False):
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
    if file_name is None:
        plt.savefig('results/estimation_visualization_scalar.png', format='png', dpi=300)
    else:
        plt.savefig(f'results/estimation_visualization_scalar_{file_name}.png', format='png', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


class TqdmPillowWriter(PillowWriter):
    def setup(self, fig, outfile, dpi, *args, **kwargs):
        super().setup(fig, outfile, dpi, *args, **kwargs)
        if isinstance(self._frames, list):
            self.total_frames = len(self._frames)
        else:
            self.total_frames = int(self._frames)
        self.pbar = tqdm(total=self.total_frames)

    def finish(self):
        super().finish()
        self.pbar.close()

    def grab_frame(self, **savefig_kwargs):
        frame = super().grab_frame(**savefig_kwargs)
        self.pbar.update(1)
        return frame

def visualize_predictions_polar(model, X_test, y_test, n_samples=9, input_window_length=400, stride=10, file_name=None, show=False):
    if len(X_test) < n_samples:
        n_samples = len(X_test)
    sample_indices = random.sample(range(len(X_test)), n_samples)

    n_cols = int(np.sqrt(n_samples))
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    axes = axes.flatten()

    def init():
        for ax in axes:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.grid(True)
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
            ax.add_patch(circle)
            ax.plot([], [], 'ro', markersize=5, label='Predicted')
            ax.plot([], [], 'go', markersize=5, label='Actual')
            ax.plot([], [], 'r-', alpha=0.5)  # Line for predicted
            ax.plot([], [], 'g-', alpha=0.5)  # Line for actual
        return [line for ax in axes for line in ax.lines]

    def animate(frame):
        updates = []
        for i, idx in enumerate(sample_indices):
            sequence = X_test[idx]
            target_sequence = y_test[idx]

            mid_point = len(sequence) // 2
            start_point = max(0, mid_point - 900)
            end_point = min(len(sequence), mid_point + 500)

            if frame < (end_point - start_point) // stride:
                start = start_point + frame * stride
                end = start + input_window_length
                if end > end_point:
                    continue

                window = sequence[start:end].reshape(1, input_window_length, -1)
                _, _, pred = model.get_X_preds(window)
                pred = np.array(pred)

                pred_x, pred_y = pred[0, 0], pred[0, 1]
                actual_x, actual_y = target_sequence[end-1][0], target_sequence[end-1][1]

                # Update positions of points and lines
                axes[i].lines[0].set_data(pred_x, pred_y)
                axes[i].lines[1].set_data(actual_x, actual_y)
                axes[i].lines[2].set_data([0, pred_x], [0, pred_y])
                axes[i].lines[3].set_data([0, actual_x], [0, actual_y])
                updates.extend(axes[i].lines)
        return updates

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=(1000 // stride), blit=False, repeat=False)
    writer = TqdmPillowWriter(fps=10)
    if file_name is None:
        ani.save(f'results/estimation_visualization_polar.gif', writer=writer)
    else:
        ani.save(f'results/estimation_visualization_polar_{file_name}.gif', writer=writer)
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()


def visualize_reference_trajectory(control_target, original_datasets, gp):
    plt.figure(figsize=(12, 8))
    plt.plot(original_datasets['header'], original_datasets[control_target], label='Actual Data (y)')
    plt.plot(original_datasets['header'], gp.y_pred, label='Predicted Data (y_pred)')
    plt.fill_between(original_datasets['header'], gp.y_pred - gp.sigma, gp.y_pred + gp.sigma,
                    color='blue', alpha=0.2, label='Confidence Interval (1 std dev)')
    plt.title('Reference trajectory')
    plt.legend(loc='upper right')
    plt.savefig(f'results/reference trajectory visualization_{control_target}.png', format='png', dpi=300)
    plt.show()

def visulize_system(idx, control_target, original, corrected, reference, scale, name="contorl", last=False):
    plt.figure(figsize=(12, 8))
    plt.plot(original['header'][:idx], original[control_target][:idx], color='black', label='original trajectory')
    plt.plot(corrected['header'], corrected[control_target], color='green', linestyle='-', linewidth=2, label='corrected trajectory')
    scale_x, scale_y = scale
    x_reference, y_reference = reference.scale(corrected['heelstrike'][-1], corrected['header'][-1], scale_x, scale_y)
    plt.plot(x_reference, y_reference, color='red', label='reference trajectory')
    x_reference2, y_reference2 = reference.scale(corrected['heelstrike'][-1], corrected['header'][-1], scale_x, scale_y, scale_from_end=False)
    half_len = int(len(x_reference2) / 4)
    plt.plot(x_reference2[:half_len], y_reference2[:half_len], color='red', linestyle='--', label='control trajectory')
    plt.xlim([min(x_reference), max(x_reference2[:half_len])])
    plt.xlim([corrected['header'][-1] - 1.2, corrected['header'][-1] + 0.2])
    if control_target == "hip_sagittal":
        plt.title("Control Hip angle")
    else:
        plt.title("Control Knee angle")
    #plt.title(f"estimaed Gait phase: {round(corrected['heelstrike'][-1], 2)} / True heelstrike: {round(original['heelstrike'][idx-1], 2)}")
    plt.legend(loc='upper right')
    plt.xlabel('time (sec)')
    plt.ylabel(f'{control_target} (deg)')
    if not os.path.exists("results/tmp"):
        os.makedirs("results/tmp")
    filename = f"results/tmp/{idx}.png"
    plt.savefig(filename)
    plt.close()
    if last:
        duration_rate = 1
        exportname = f"results/{name}"
        frames = []
        png_files = sorted(glob.glob(os.path.join('results/tmp', '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        for png_file in png_files:
            frames.append(imageio.imread(png_file))
        imageio.mimsave(f"{exportname}.gif", frames, format='GIF', duration=duration_rate)

        # for filename in set(png_files):
        #     os.remove(filename)