import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import tf_util
import load_policy
from sklearn.model_selection import train_test_split
from callbacks import CloneCallback
from beh_cloning import *
def dagger():
	with tf.Session():
		tf_util.initialize()
		batch_size = 64
		parser = argparse.ArgumentParser()
		parser.add_argument('expert_policy_file', type=str)
		parser.add_argument('data_path', type=str)
		parser.add_argument('env_name', type=str)
		parser.add_argument('num_rollouts', type=int)
		parser.add_argument('num_iterations', type=int)
		args = parser.parse_args()
		env = gym.make(args.env_name)
		data = load_data(args.data_path)
		x_train_original, x_test_original, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
		policy_fn = load_policy.load_policy(args.expert_policy_file)
		dagger_reward_means = []
		dagger_reward_stds = []
		for _ in range(args.num_iterations):
			x_train_mean, x_train_std = x_train_original.mean(axis=0), x_train_original.std(axis=0)
			x_train_std[np.where(x_train_std == 0)] = np.random.normal(0, 0.1, 1) * 10e-10
			x_train = (x_train_original - x_train_mean) / x_train_std
			x_test = (x_test_original - x_train_mean) / x_train_std
			csv_logger = tf.keras.callbacks.CSVLogger('./logs/csv/dagger-{}-{}'.format(args.env_name, args.num_rollouts))
			clone_callback = CloneCallback(gym.make(args.env_name), x_train_mean, x_train_std, args.num_rollouts)
			callbacks = [csv_logger, clone_callback]
			odim = x_train[0].shape[0]
			adim = y_train[0].shape[1]
			model = create_model(odim, adim)
			model.compile(loss='mse', optimizer='rmsprop')
			print("LENGTH" + str(x_train.shape))
			hist = model.fit_generator(
				          generator(x_train, y_train, batch_size),
						  callbacks=callbacks,
						  epochs=10,
						  steps_per_epoch=len(x_train)/batch_size,
						  validation_data=generator(x_test, y_test, batch_size),
						  validation_steps=len(x_test))
			dagger_reward_means.append(clone_callback.reward_means[len(clone_callback.reward_means) - 1])
			dagger_reward_stds.append(clone_callback.stds[len(clone_callback.stds) - 1])
			print(clone_callback.reward_means)
			observations = []
			expert_actions = []
			max_steps = env.spec.timestep_limit
			steps = 0
			obs = env.reset()
			done = False
			for _ in range(100):
				while not done:
					observations.append(obs)
					expert_actions.append(policy_fn(obs[None,:]))
					obs = (obs - x_train_mean) / x_train_std
					action = model.predict(np.reshape(np.array(obs), (1, len(obs))))
					try:
						obs, r, done, _ = env.step(action)
						steps += 1
					except:
						steps += 1
					if steps >= max_steps:
						break
			print("OBSERVATION SHAPE" + str(np.array(observations).shape))				
			print("ACTION SHAPE" + str(np.array(expert_actions).shape))
			x_train_original = np.vstack([x_train_original, np.array(observations)])
			y_train = np.vstack([y_train, np.array(expert_actions)])
		with open("./logs/average_rewards/dagger-{}-{}.pkl".format(args.env_name, args.num_rollouts), 'wb') as f:
			pickle.dump({'avg_reward': np.array(dagger_reward_means), 'std': np.array(dagger_reward_stds)}, f)



if __name__ == '__main__':
	dagger()