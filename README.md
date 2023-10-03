# "CSE571 AI-based Mobile Robotics"
### University of Washington Graduate Course
The course covered theoretically far more, but the following methods/algorithms were implemented in practice: 
- [**Localization**](StateEstimation): _EKF / Particle Filter_ (Odometry Motion Model, Landmark Sensor Model) on a car in [PyBullet](https://pybullet.org/wordpress/). 
- [**Motion Planning**](MotionPlanning): _A* / LPA* / D* / RRT / RRT*_ on 2-DoF/3-DoF robotic arms with scene obstacles, RRT* on a non-holonomic car in PyBullet. 
- [**Reinforcement Learning**](ReinforcementLearning): _Behavior Cloning / DAgger_ on [Reacher](https://www.gymlibrary.dev/environments/mujoco/reacher/), _Policy Gradient_ on [Inverted Pendulum](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/), all of which are parts of [MuJoCo environment](https://www.gymlibrary.dev/environments/mujoco/inverted_pendulum/).

This repo content discusses some observations and analyses of the methods' performance based on different parameters, reinforced with visual proofs (plots, screenshots, GIFs)

***The code for the methods is NOT provided due to the course policy***
