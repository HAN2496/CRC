# Machine Learning Applications for Lower Limb Exoskeleton Robots

&nbsp;

## Description
This repository contains the machine learning algorithms developed for enhancing the rehabilitation process of children with cerebral palsy (CP) using lower limb exoskeleton robots. The project focuses on creating adaptive control systems that adjust in real-time to the user's gait, step length, and walking speed to improve walking patterns more effectively.

&nbsp;

## Features
- Real-time gait phase estimation using time-series data.
- Adaptive control trajectory generation based on machine learning predictions.
- Use of Gaussian process regression for generating reference trajectories.
- Simulation environment setup for testing with actual data.
  
&nbsp;

## Installation
To set up this project along with the necessary dependencies, follow these steps:

### Prerequisites
Ensure that you have Python installed on your system. This project requires:
- Python 3.8.10

### Clone the repository
```bash
git clone https://github.com/manhyeongwoo/MLP_project_4.git
cd MLP_project_4
```

### Install the required packages
```bash
pip install -r requirements.txt
```

### Install the tsai library
```bash
pip install tsai
```
For more information on the tsai library, visit the [tsai GitHub repository](https://github.com/timeseriesAI/tsai).

&nbsp;

## Usage
To run the simulation, execute the following command:
```bash
# For training estimation model
python -m train

# For simulation
python -m simulation

# For Gaussian process regression
python -m src/trajectory_generator

```
&nbsp;

## Results
Here are some key outcomes from our simulations:

&nbsp;

**Gait Phase Estimation**: Our model demonstrates high accuracy in estimating the gait phase, helping to tailor the therapy to individual needs. This section can be enhanced with a graph showing accuracy over time.
<table width="100%">
  <tr>
    <td width="45%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/719780e0-4ccf-4ae4-ba93-c7f05dfaafac" width="100%"></td>
    <td width="55%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/c585e3f2-2962-4281-a6b8-7f65e0b25105" width="100%"></td>
  </tr>
  <tr>
    <td width="45%" align="center">Vector Visualization</td>
    <td width="55%" align="center">Scalar Visualization</td>
  </tr>
</table>
&nbsp;

**Gaussian Process Regression for Joint Angles**: We applied Gaussian process regression to model the typical joint trajectories for hips and knees. The results helped us understand variations and predict movements more accurately.
<table width="100%">
  <tr>
    <td width="50%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/a3f07bc7-24ae-436b-8745-7de4e07074a2" width="100%"></td>
    <td width="50%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/9632b25b-e188-4703-916a-2db4e6127374" width="100%"></td>
  </tr>
  <tr>
    <td width="50%" align="center">Reference Trajectory of Hip Joint</td>
    <td width="50%" align="center">Reference Trajectory of Knee Joint</td>
  </tr>
</table>
&nbsp;

**Control Simulation**: The control simulations for hip and knee joints demonstrate the effectiveness of our adaptive algorithms in real-time corrections, significantly improving alignment with the desired trajectories.
<table width="100%">
  <tr>
    <td width="50%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/83323eaa-a5d7-41e2-8eaa-7790afb69a6d" width="100%"></td>
    <td width="50%"><img src="https://github.com/namhyeongwoo/CRC_tsai/assets/88234001/5eaa7e42-ba2a-476c-b58d-0e5748c10563" width="100%"></td>
  </tr>
  <tr>
    <td width="50%" align="center">Control Simulation of Hip Joint Angle</td>
    <td width="50%" align="center">Control Simulation of Knee Joint Angle</td>
  </tr>
</table>
&nbsp;

These results illustrate the effectiveness of our model in providing adaptive, personalized rehabilitation interventions.
