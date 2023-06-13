# Autonomous Car
This project is the extracurricular sequel of "CSE571 AI-based Mobile Robotics" course. 
The course covered far more, but in assignments the following methods/algorithms were implemented: 
*Localization*: **EKF / Particle Filter** (Odometry Motion Model, Landmark Sensor Model) on a car in PyBullet. 
_Motion Planning_: __A* / LPA* / D* / RRT / RRT*__ on 2-DoF/3-DoF robotic arm with scene obstacles, RRT* for non-holonomic car in PyBullet. 
_Reinforcement Learning_: __Behavior Cloning / DAgger__ on [Reacher](https://www.gymlibrary.dev/environments/mujoco/reacher/), **Policy Gradient** on [Inverted Pendulum](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/) all of which are parts of [MuJoCo environment](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/). 

The goal of this project is to analyze performance of each algorithm (Localization, Motion Planning) on [MuSHR](https://mushr.io/) car in ROS simulation, and implement D* motion planning, Particle Filter with Landmark Sensor Model (using Intel Realsense Depth Camera in MuSHR car) on the real car. 
D* is notorious for its capability to quickly replan the path from a current robot's state to a changing goal state in dynamic environment. 
Particle Filter requires less computation power than EKF and is flexible enough to face non-linearity. Landmark Sensor Model is prefered over Laser Sensor Model, which is implemented for the real MuSHR car in the [EEP545_Self-Driving_Cars](https://github.com/SanjarNormuradov/EEP545_Self-Driving_Cars) project, because of novelty to the author: Computer Vision and Deep Learning to train the model. 
