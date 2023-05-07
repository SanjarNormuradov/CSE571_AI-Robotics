import numpy as np

import utils 


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u_noisefree):
        """
        Update particles after taking an action

        self.particles: np.array(self.num_particles, 3) - matrix of poses (x,y,theta)
        u_noisefree: [rot1, trans, rot2].T - noisefree action
        """
        new_particles = self.particles

        # Add noise to action:
        #   rot1' = rot1 + Qrot1
        #   trans' = trans + Qtrans
        #   rot2' = rot2 + Qrot2
        # Where: (note usage of self.alphas = filter.factor * init_alphas and alphas = data.factor * init_alphas in env.noise_from_motion() function)
        #   Qrot1 = self.alphas[0] * rot1**2 + self.alphas[1] * trans**2
        #   Qtrans = self.alphas[2] * trans**2 + self.alphas[3] * (rot1**2 + rot2**2)
        #   Qrot2 = self.alphas[0] * rot2**2 + self.alphas[1] * trans**2
        # 
        # u_noisy: [rot1', trans', rot2'].T - noisy action
        for i in range(self.num_particles):
            u_noisy = env.sample_noisy_action(u_noisefree, self.alphas)
            theta = new_particles[i, 2] + u_noisy[0]
            new_particles[i, 0] += u_noisy[1] * np.cos(theta)
            new_particles[i, 1] += u_noisy[1] * np.sin(theta)
            new_particles[i, 2] = utils.minimized_angle(theta + u_noisy[2])
        return new_particles

    def update(self, env, u_noisefree, z_real, marker_id):
        """
        Update the state estimate after taking an action and receiving a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # Prediction
        particles = self.move_particles(env, u_noisefree)

        # Correction
        weights = self.weights
        for i in range(self.num_particles):
            # Update importance weights depending on the likelihood of particle i measurement to be real observation (z_real)   
            weights[i] = env.likelihood(z_real - env.observe(particles[i], marker_id), self.beta)
        #   Normalize weights
        if np.array_equal(weights, np.zeros(self.num_particles)):
            weights[:] = np.ones(self.num_particles) / self.num_particles
        else:
            weights[:] /= weights.sum()

        # Resampling
        self.particles = self.resample(particles, weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """
        Sample new particles and weights given current particles and weights. 
        Be sure to use the low-variance sampler from class.

        particles: np.array(n, 3) - matrix of poses (x,y,theta)
        weights: np.array(n,)
        """
        M = self.num_particles
        new_particles = np.zeros(particles.shape)
        r = np.random.uniform(0.0, 1.0 / M)
        c = weights[0]
        i, j = 0, 0
        for m in range(M):
            u = r + m * 1.0 / M
            while u > c:
                i += 1
                c += weights[i]
            new_particles[j] = self.particles[i]
            j += 1

        particles[:] = new_particles[:]                            
        return new_particles

    def mean_and_variance(self, particles):
        """
        Compute the mean and covariance matrix for a set of equally-weighted particles.

        particles: np.array(n, 3) - matrix of poses (x,y,theta)
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum(),
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = utils.minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(particles.shape[1]) * 1e-6  # Avoid bad conditioning 

        return mean.reshape((-1, 1)), cov
