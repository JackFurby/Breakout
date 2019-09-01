from model import Model
import tensorflow as tf
import numpy as np
import random


class Agent:
	def __init__(self, width, height, actions, log_dir):
		self.learningRate = 0.001
		self.replayMemorySize = 10_000  # Number of states that are kept for training
		self.minReplayMemSize = 2_000  # Min size of replay memory before training starts
		self.batchSize = 32  # How many samples are used for training
		self.updateEvery = 10  # Number of batches between updating target network
		self.discount = 0.99  # messure of how much we care about future reward over immedate reward
		self.actions = actions

		self.model = Model(width, height, self.actions)  # model for training
		self.model.compile(
			#optimizer=tf.keras.optimizers.Adam(lr=self.learningRate),
			optimizer=tf.keras.optimizers.RMSprop(lr=self.learningRate, rho=0.95, epsilon=0.01),
			loss=tf.keras.losses.MeanSquaredError(),
			#loss=tf.keras.losses.Huber(),
			metrics=['accuracy']
		)
		self.targetModel = Model(width, height, self.actions)  # model for predictions
		self.targetModel.set_weights(self.model.get_weights())
		self.replayMemory = []  # last n steps
		self.targetUpdateCounter = 0  # counter since last target model update

	# Queries main network for Q values given current state
	def get_qs(self, state):
		return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

	# Add new step to replayMemory. ReplayMemory is a first in first out list
	def update_replay_memory(self, transition):
		if len(self.replayMemory) >= self.replayMemorySize:
			self.replayMemory.pop(0)
			self.replayMemory.append(transition)
		else:
			self.replayMemory.append(transition)

	# Trains main network every step during game
	def train(self, terminal_state, step):

		# Start training only if certain number of samples is already saved
		if len(self.replayMemory) < self.minReplayMemSize:
			return

		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replayMemory, self.batchSize)

		# Get current states from minibatch, then query NN model for Q values
		currentStates = np.array([transition[0] for transition in minibatch])
		#currentQsList = self.model.predict(currentStates)

		# Get future states from minibatch, then query NN model for Q values
		newCurrentStates = np.array([transition[3] for transition in minibatch])
		futureQsList = self.targetModel.predict(newCurrentStates)

		#batchActions = np.zeros((self.batchSize, self.actions))
		batchTargets = np.zeros((self.batchSize, self.actions))

		#X = []
		y = []

		# Enumerate minibatch to prepare for fitting
		for index, (currentState, action, reward, newCurrentState, done) in enumerate(minibatch):

			# If not a terminal state, get new q from future states, otherwise set it to 0
			if not done:
				max_future_q = np.max(futureQsList[index])
				newQ = reward + (self.discount * max_future_q)
			else:
				newQ = reward

			# Update Q value for given state
			#currentQs = currentQsList[index]
			action[np.argmax(action)] = newQ

			#batchActions[index][action] = 1
			batchTargets[index][np.argmax(action)] = 1

			# And append to our training data
			#X.append(currentState)
			y.append(action)

		# Fit on all minibatch and return loss and accuracy
		#metrics = self.model.fit(np.array(X), np.array(y), batch_size=self.batchSize, verbose=0, shuffle=False)
		metrics = self.model.fit(currentStates, batchTargets, batch_size=self.batchSize, verbose=0, shuffle=False)

		# Update target network counter every game
		if terminal_state:
			self.targetUpdateCounter += 1

		# If counter reaches set value, update target network with weights of main network
		if self.targetUpdateCounter > self.updateEvery:
			self.targetModel.set_weights(self.model.get_weights())
			self.targetUpdateCounter = 0

		return metrics
