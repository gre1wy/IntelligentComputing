import numpy as np

class Particle:
    def __init__(self, dim, pos_bounds, vel_bounds):
        """
        Initializes a particle with random position and velocity.
        
        Parameters:
            dim (int): Dimensionality of the search space.
            pos_bounds (tuple): (x_min, x_max) boundaries for positions.
            vel_bounds (tuple): (v_min, v_max) boundaries for velocities.
        """
        self.dim = dim
        self.pos_bounds = np.tile(np.array(pos_bounds), (dim, 1)) if np.array(pos_bounds).ndim == 1 else np.array(pos_bounds)
        self.vel_bounds = np.tile(np.array(vel_bounds), (dim, 1)) if np.array(vel_bounds).ndim == 1 else np.array(vel_bounds)
        self.position = np.random.uniform(self.pos_bounds[:, 0], self.pos_bounds[:, 1])
        self.velocity = np.random.uniform(self.vel_bounds[:, 0], self.vel_bounds[:, 1])
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best, w, alpha1, alpha2):
        """
        Updates the particle's velocity based on inertia, cognitive, and social components.
        
        Parameters:
            global_best (ndarray): The best position found by the swarm.
            w (float): Inertia weight.
            alpha1 (float): Cognitive acceleration coefficient.
            alpha2 (float): Social acceleration coefficient.
        """
        r1, r2 = np.random.rand(2)
        cognitive = alpha1 * r1 * (self.best_position - self.position)
        social = alpha2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, self.vel_bounds[:, 0], self.vel_bounds[:, 1])


    def update_position(self):
        """
        Updates the particle's position based on its velocity and applies boundary reflection.
        """
        self.position += self.velocity
        # Reflect the particle if it goes out of bounds
        for j in range(self.dim):
            if self.position[j] < self.pos_bounds[j, 0]:
                self.position[j] = self.pos_bounds[j, 0] + abs(self.position[j] - self.pos_bounds[j, 0])
                self.velocity[j] = -self.velocity[j]
            elif self.position[j] > self.pos_bounds[j, 1]:
                self.position[j] = self.pos_bounds[j, 1] - abs(self.position[j] - self.pos_bounds[j, 1])
                self.velocity[j] = -self.velocity[j]

class PSO:
    def __init__(self, fitness_func, num_particles, dim, pos_bounds, vel_bounds, max_iter, 
                 alpha1, alpha2, w_max, w_min, 
                 enable_shaking=False, stagnation_iter=20, shake_probability=0.5, shake_amplitude=0.1,
                 target_global_min=None, epsilon_target=None, epsilon_stagnation=1e-6):
        self.fitness_func = fitness_func
        self.num_particles = num_particles
        self.dim = dim
        self.pos_bounds = pos_bounds
        self.vel_bounds = vel_bounds
        self.max_iter = max_iter
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.w_max = w_max
        self.w_min = w_min

        self.target_global_min = target_global_min
        self.epsilon_target = epsilon_target

        self.enable_shaking = enable_shaking
        self.epsilon_stagnation = epsilon_stagnation
        self.stagnation_iter = stagnation_iter
        self.shake_probability = shake_probability
        self.shake_amplitude = shake_amplitude

        self.swarm = [Particle(dim, pos_bounds, vel_bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')

        self.history_best = []
        self.history_all_positions = []

        self._initialize_global_best()

    def _initialize_global_best(self):
        for particle in self.swarm:
            score = self.fitness_func(particle.position)
            particle.best_score = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()

    def _update_particles(self, w):
        for particle in self.swarm:
            particle.update_velocity(self.global_best_position, w, self.alpha1, self.alpha2)
            particle.update_position()
            score = self.fitness_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()

    def _record_history(self):
        """
        Saves the global best position and its score for the current iteration,
        and all particles' positions.
        """
        self.history_best.append((self.global_best_position.copy(), self.global_best_score))

        iteration_positions = [p.position.copy() for p in self.swarm]
        self.history_all_positions.append(iteration_positions)


    def _check_stagnation(self, previous_best, stagnation_count):
        if abs(previous_best - self.global_best_score) < self.epsilon_stagnation:
            return stagnation_count + 1, previous_best
        return 0, self.global_best_score

    def _check_termination(self, iteration):
        if self.target_global_min is not None:
            if self.global_best_score <= self.target_global_min:
                print(f"Target global minimum {self.target_global_min} reached at iteration {iteration}. Stopping.")
                return True
            if self.epsilon_target is not None and abs(self.global_best_score - self.target_global_min) < self.epsilon_target:
                print(f"Global minimum is within epsilon {self.epsilon_target} of target at iteration {iteration}. Stopping.")
                return True
        return False

    def shake_particles(self):
        range_pos = np.abs(self.pos_bounds[1] - self.pos_bounds[0])
        shake_std = np.maximum(self.shake_amplitude * range_pos, 1e-6)

        shaken_count = 0
        for particle in self.swarm:
            if np.random.rand() < self.shake_probability:
                noise = np.random.normal(0, shake_std, self.dim)
                particle.position += noise
                particle.velocity = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], self.dim)
                particle.best_position = particle.position.copy()
                particle.best_score = self.fitness_func(particle.position)
                shaken_count += 1
        print(f"Shaken {shaken_count} particles.")

    def run(self):
        stagnation_count = 0
        previous_best = self.global_best_score

        for iteration in range(self.max_iter):
            w = self.w_max - ((self.w_max - self.w_min) * iteration / self.max_iter)

            self._update_particles(w)
            self._record_history()
            stagnation_count, previous_best = self._check_stagnation(previous_best, stagnation_count)

            if self.enable_shaking and stagnation_count >= self.stagnation_iter:
                print(f"Stagnation reached at iteration {iteration}. Shaking particles...")
                self.shake_particles()
                stagnation_count = 0

            if self._check_termination(iteration):
                break

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Global best score = {self.global_best_score}")

        return self.global_best_position, self.global_best_score, self.history_best, self.history_all_positions
