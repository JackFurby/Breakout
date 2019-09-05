import numpy as np
import os
import cv2
import gym
import tensorflow as tf
from agent import Agent
import time

env = gym.make('BreakoutDeterministic-v4')

SAMPLE_WIDTH = 84
SAMPLE_HEIGHT = 84

LATEST_WEIGHTS = tf.train.latest_checkpoint('checkpoints')

# Convert image to greyscale, resize and normalise pixels
def preprocess(screen, width, height, targetWidth, targetHeight):
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
	screen = screen[20:300, 0:200]  # crop off score
	screen = cv2.resize(screen, (targetWidth, targetHeight))
	screen = screen.reshape(targetWidth, targetHeight) / 255
	return screen

agent = Agent(
	width=SAMPLE_WIDTH,
	height=SAMPLE_HEIGHT,
	actions=env.action_space.n
)

agent.model.load_weights(LATEST_WEIGHTS)

while True:

	# Reset environment and get initial state
	current_state = env.reset()
	current_state = preprocess(
		current_state,
		env.observation_space.shape[0],
		env.observation_space.shape[1],
		SAMPLE_WIDTH,
		SAMPLE_HEIGHT
	)
	current_state = np.dstack((current_state, current_state, current_state, current_state))

	# Reset flag and start iterating until episode ends
	done = False
	while not done:

		action = np.argmax(agent.get_qs(current_state))

		new_state, reward, done, info = env.step(action)

		new_state = preprocess(
			new_state,
			env.observation_space.shape[0],
			env.observation_space.shape[1],
			SAMPLE_WIDTH,
			SAMPLE_HEIGHT
		)

		new_state = np.dstack((new_state, current_state[:, :, 0], current_state[:, :, 1], current_state[:, :, 2]))

		env.render()

		current_state = new_state
		time.sleep(1 / 30)  # lock framerate to aprox 30 fps
