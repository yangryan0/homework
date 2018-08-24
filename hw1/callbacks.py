import gym
import numpy as np
import tensorflow as tf

class CloneCallback(tf.keras.callbacks.Callback):
	def __init__(self, envname, mean, std, num_rollouts):
		self.env = gym.make(envname)
		self.num_rollouts = num_rollouts
		self.mean = mean
		self.std = std
		self.reward_means = []
		self.stds = []
		self.max_steps = self.env.spec.timestep_limit

	def on_epoch_end(self, epoch, logs={}):
		rewards = []
		for i in range(self.num_rollouts):
			done = False
			totalr = 0
			steps = 0
			obs = self.env.reset()
			while not done:
				obs = (obs - self.mean) / self.std
				action = self.model.predict(np.reshape(np.array(obs), (1, len(obs))))
				try:
					obs, r, done, _ = self.env.step(action)
				except:
					steps += 1
					pass
				totalr += r
				steps += 1
				if steps >= self.max_steps:
					break
			rewards.append(totalr)
		self.reward_means.append(np.mean(rewards))
		self.stds.append(np.std(rewards))
