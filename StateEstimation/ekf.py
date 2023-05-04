import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u_noisefree, z_real, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u_noisefree: [rot1, trans, rot2].T - noisefree action
        z_real: [phi,].T - real landmark observation, i.e. angle between robot's previous heading angle self.mu[2] and straight line connected to the landmark marker_id
        marker_id: landmark ID
        """
        # Linearize dynamics (noisefree gradient computation)
        Gmat = env.G(self.mu, u_noisefree)
        Vmat = env.V(self.mu, u_noisefree)

        # Add noise to action:
        #   rot1' = rot1 + Qrot1
        #   trans' = trans + Qtrans
        #   rot2' = rot2 + Qrot2
        # Where: (note usage of self.alphas = filter.factor * init_alphas and alphas = data.factor * init_alphas in env.noise_from_motion() function)
        #   Qrot1 = self.alphas[0] * rot1**2 + self.alphas[1] * trans**2
        #   Qtrans = self.alphas[2] * trans**2 + self.alphas[3] * (rot1**2 + rot2**2)
        #   Qrot2 = self.alphas[0] * rot2**2 + self.alphas[1] * trans**2
        # 
        # Mmat = [[Qrot1, 0, 0], [0, Qtrans, 0], [0, 0, Qrot2]]
        # u_noisy: [rot1', trans', rot2'].T - noisy action
        Mmat = env.noise_from_motion(u_noisefree, self.alphas)
        u_noisy = env.sample_noisy_action(u_noisefree, self.alphas) 

        # Prediction:
        #   Mean:
        #       x' = x + trans' * cos(theta + rot1')
        #       y' = y + trans' * sin(theta + rot1')
        #       theta' = theta + rot1' + rot2'
        self.mu[0] += u_noisy[1] * np.cos(self.mu[2] + u_noisy[0])
        self.mu[1] += u_noisy[1] * np.sin(self.mu[2] + u_noisy[0])
        self.mu[2] += u_noisy[0] + u_noisy[2]
        #   Covariance:
        #       cov' = Gmat * cov * Gmat.T + Vmat * Mmat * Vmat.T
        self.sigma = Gmat.dot(self.sigma).dot(Gmat.T) + Vmat.dot(Mmat).dot(Vmat.T)

        # Linearize measurement
        # Hmat shape: (1,3)
        Hmat = env.H(self.mu, marker_id)

        # Correction
        #   Kalman Gain:
        #       K = cov' * Hmat.T * (Hmat * cov' * Hmat.T + self.beta)^(-1)
        K = self.sigma.dot(Hmat.T) * (np.linalg.inv((Hmat.dot(self.sigma).dot(Hmat.T) + self.beta)))
        #   Mean:
        #       self.mu = self.mu + K * (z_real - h(self.mu))
        self.mu += K * (z_real - env.observe(self.mu, marker_id))
        #   Covariance:
        #       cov" = (I - K * Hmat) * cov'
        self.sigma = (np.eye(3) - K.dot(Hmat)).dot(self.sigma)

        return self.mu, self.sigma
