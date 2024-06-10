from manim import *

class GradientDescentPlot(Scene):
    def construct(self):
        # Initialize graph components
        original_data = self.create_dot_plot(original_datas['header'], original_datas['hip_sagittal'], color=RED, label="Actual Data")
        corrected_data = self.create_line_plot(corrected_datas['header'], corrected_datas['hip_sagittal'], color=BLUE, label="Corrected Subject")
        reference_traj = self.create_line_plot([], [], color=GREEN, label="Reference Trajectory")
        control_traj = self.create_line_plot([], [], color=BLACK, label="Control Trajectory")

        title = Text("")
        
        self.add(original_data, corrected_data, reference_traj, control_traj, title)

        for i in range(control.start_idx, end_idx):
            self.update_frame(i, original_data, corrected_data, reference_traj, control_traj, title)
            self.wait(0.1)  # Adjust speed of the animation

    def create_dot_plot(self, x_data, y_data, color=RED, label=""):
        return VGroup(*[Dot(point=[x, y, 0], color=color) for x, y in zip(x_data, y_data)])

    def create_line_plot(self, x_data, y_data, color=RED, label=""):
        return VGroup(*[Line(start=[x_data[i-1], y_data[i-1], 0], end=[x, y, 0], color=color) for i, (x, y) in enumerate(zip(x_data, y_data)) if i > 0])

    def update_frame(self, i, original_data, corrected_data, reference_traj, control_traj, title):
        idx, heelstrike, header, detail_scale_histories = scale_histories[i]
        scale_x, scale_y = detail_scale_histories
        X_scalar_pred_new, y_pred_new = gp.scale(heelstrike, scale_x, scale_y, header)
        X_scalar_pred_new2, y_pred_new2 = gp.scale(heelstrike, scale_x, scale_y, header, end=False)

        # Update plots
        reference_traj.become(self.create_line_plot(X_scalar_pred_new, y_pred_new, color=GREEN))
        control_traj.become(self.create_line_plot(X_scalar_pred_new2, y_pred_new2, color=BLACK))
        title.become(Text(f'Iteration {i}: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}', font_size=24))

# To render the scene, use the following command in your terminal:
# manim -pql your_script.py GradientDescentPlot
