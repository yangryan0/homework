import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import tf_util
import load_policy
from sklearn.model_selection import train_test_split
from callbacks import CloneCallback

def create_model(input_dim, output_dim): 
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.InputLayer(batch_input_shape=(None, input_dim,)))
	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dense(output_dim))
	return model

def load_data(path):
	with open(path, 'rb') as f:
		data = pickle.loads(f.read())
	return data['observations'], data['actions']

def generator(x_train, y_train, batch_size):
	while True:
		indices = np.random.choice(len(x_train), batch_size)
		observations, actions = x_train[indices], y_train[indices]
		actions = [action.flatten() for action in actions]
		yield np.asarray(observations), np.asarray(actions)

def main():
	batch_size = 64
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', type=str)
	parser.add_argument('env_name', type=str)
	parser.add_argument('num_rollouts', type=int)
	args = parser.parse_args()
	data = load_data(args.data_path)
	x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2)
	x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
	x_train_std[np.where(x_train_std == 0)] = np.random.normal(0, 0.1, 1) * 10e-10
	x_train = (x_train - x_train_mean) / x_train_std
	x_test = (x_test - x_train_mean) / x_train_std
	csv_logger = tf.keras.callbacks.CSVLogger('./logs/csv/{}-{}'.format(args.env_name, args.num_rollouts))
	clone_callback = CloneCallback(gym.make(args.env_name), x_train_mean, x_train_std, args.num_rollouts)
	odim = x_train[0].shape[0]
	adim = y_train[0].shape[1]
	model = create_model(odim, adim)
	model.compile(loss='mse', optimizer='rmsprop')
	callbacks = [csv_logger, clone_callback]
	print("LENGTH" + str(len(x_train)))
	hist = model.fit_generator(
		          generator(x_train, y_train, batch_size),
				  callbacks=callbacks,
				  epochs=20,
				  steps_per_epoch=len(x_train)/batch_size,
				  validation_data=generator(x_test, y_test, batch_size),
				  validation_steps=len(x_test))
	with open("./logs/average_rewards/avg-{}-{}.pkl".format(args.env_name, args.num_rollouts), 'wb') as f:
		pickle.dump({'avg_reward': np.array(clone_callback.reward_means), 'std': np.array(clone_callback.stds)}, f)

if __name__ == '__main__':
	main()