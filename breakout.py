import numpy as np
import random
from tqdm import tqdm
import os
import datetime
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from agent import Agent

# Environment settings
EPISODES = 1500

# Exploration settings
#epsilon = 1  # starting epsolon
epsilon = 0.1  # starting epsolon
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.1

#  Stats settings
SHOW_PREVIEW = True
RENDER_PREVIEW = 5  # render every x episodes

env = gym.make('BreakoutDeterministic-v4')

SAMPLE_WIDTH = 84
SAMPLE_HEIGHT = 84

MODEL_NAME = '16x32-'


# Convert image to greyscale, resize and normalise pixels
def preprocess(screen, width, height, targetWidth, targetHeight):
	# plt.imshow(screen)
	# plt.show()
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
	screen = screen[20:300, 0:200]  # crop off score
	screen = cv2.resize(screen, (targetWidth, targetHeight))
	screen = screen.reshape(targetWidth, targetHeight) / 255
	# plt.imshow(np.array(np.squeeze(screen)), cmap='gray')
	# plt.show()
	return screen


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + MODEL_NAME + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
checkpoint_path = "checkpoints/cp-{episode:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

agent = Agent(
	width=SAMPLE_WIDTH,
	height=SAMPLE_HEIGHT,
	actions=env.action_space.n,
	log_dir=train_log_dir
)

agent.model.load_weights(tf.train.latest_checkpoint('checkpoints'))  # Uncomment to load from checkpoint

average_reward = []

# Iterate over episodes
for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):

	average_loss = []
	average_accuracy = []
	episode_reward = 0
	step = 1

	# Reset environment and get initial state
	current_state = env.reset()
	current_state = preprocess(
		current_state,
		env.observation_space.shape[0],
		env.observation_space.shape[1],
		SAMPLE_WIDTH,
		SAMPLE_HEIGHT
	)
	currentLives = 5  # starting lives for episode

	current_state = np.dstack((current_state, current_state, current_state, current_state))

	# Reset flag and start iterating until episode ends
	done = False
	while not done:

		# Get action from Q table
		if np.random.random() > epsilon:
			action = np.argmax(agent.get_qs(current_state))
		# Get random action
		else:
			action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)

		episode_reward += reward

		# If life is lost then give negative reward
		if info["ale.lives"] < currentLives:
			reward = -1
			done = True

		reward = agent.clip_reward(reward)

		new_state = preprocess(
			new_state,
			env.observation_space.shape[0],
			env.observation_space.shape[1],
			SAMPLE_WIDTH,
			SAMPLE_HEIGHT
		)

		new_state = np.dstack((new_state, current_state[:, :, 0], current_state[:, :, 1], current_state[:, :, 2]))

		if SHOW_PREVIEW and episode % RENDER_PREVIEW == 0:
			env.render()

		# Every step we update replay memory and train main network
		# print(np.dstack((stateStack[0], stateStack[1], stateStack[2], stateStack[3])).shape)
		agent.update_replay_memory((current_state, agent.get_qs(current_state), reward, new_state, done))
		metrics = agent.train(done, step)

		if metrics is not None:
			average_loss.append(metrics.history['loss'][0])
			average_accuracy.append(metrics.history['accuracy'][0])

		current_state = new_state
		currentLives = info["ale.lives"]  # update lives remaining
		step += 1

	if len(average_reward) >= 5:
		average_reward.pop(0)
		average_reward.append(episode_reward)
	else:
		average_reward.append(episode_reward)

	with train_summary_writer.as_default():
		tf.summary.scalar('episode score', episode_reward, step=episode)
		tf.summary.scalar('average score', sum(average_reward) / len(average_reward), step=episode)
		tf.summary.scalar('epsilon', epsilon, step=episode)
		if len(average_loss) > 0:
			tf.summary.scalar('loss', sum(average_loss) / len(average_loss), step=episode)
		if len(average_accuracy) > 0:
			tf.summary.scalar('accuracy', sum(average_accuracy) / len(average_accuracy), step=episode)

	agent.model.save_weights(checkpoint_path.format(episode=episode))

	# Decay epsilon. Only start when replay memory is over min size
	if epsilon > MIN_EPSILON and agent.over_min_batch_size:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)

agent.model.save_weights('models/' + MODEL_NAME + current_time, save_format='tf')
