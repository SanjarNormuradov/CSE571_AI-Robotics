# Autonomous Car
This project is the extracurricular sequel of the "CSE571 AI-based Mobile Robotics" course. 

The goals of this project:
- analyze the performance of each algorithm in State Estimation/Motion Planning on [MuSHR](https://mushr.io/) car in ROS simulation
- implement on the real MuSHR car:
  - D* Motion Planning
    - D* is notorious for its capability to quickly replan the path from a current robot's state to a changing goal state in a dynamic environment.
  - Particle Filter Localization with Landmark Sensor Model
    - Particle Filter requires less computation power than EKF and is flexible enough to face non-linearity. 
    - Landmark Sensor Model is preferred over the Laser Sensor Model, which is implemented for the real MuSHR car in the [EEP545_Self-Driving_Cars](https://github.com/SanjarNormuradov/EEP545_Self-Driving_Cars) project, because of novelty to the author: Computer Vision (using Intel Realsense Depth Camera of MuSHR car) and Deep Learning to train the model. 

## Course "CSE571 AI-based Mobile Robotics"
The course covered theoretically far more, but the following methods/algorithms were implemented in practice: 
- [**Localization**](StateEstimation): _EKF / Particle Filter_ (Odometry Motion Model, Landmark Sensor Model) on a car in [PyBullet](https://pybullet.org/wordpress/). 
- [**Motion Planning**](MotionPlanning): _A* / LPA* / D* / RRT / RRT*_ on 2-DoF/3-DoF robotic arm with scene obstacles, RRT* for non-holonomic car in PyBullet. 
- [**Reinforcement Learning**](ReinforcementLearning): _Behavior Cloning / DAgger_ on [Reacher](https://www.gymlibrary.dev/environments/mujoco/reacher/), _Policy Gradient_ on [Inverted Pendulum](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/), all of which are parts of [MuJoCo environment](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/).

***The code for the assignments is NOT provided due to the course policy.***
