# Breakout

## Set-up

For this project I have implemented a CNN in Tensorflow 2.0. This is almost exactly the same as DeepMinds implementation from their paper here: https://arxiv.org/pdf/1312.5602v1.pdf. This resulted in 3 parts. this readme is in two parts. The first is a brief breakdown of the implementation and the second part will run over some of the testing I made and what I have learned.

### breakout.py

The Breakout environment is run with each frame being recorded (current state) along with an action and reward and next state. The current and next state is a combination of 4 frames which captures ball movement. Every frame has been cropped, converted to grey scale and values changed to be between 0 and 255. breakout.py runs the environment among with adjusting epsilon, start training, recording logs and saving the model.

### agent.py

The agent is formed of a number of adjustable variables, two neural networks and replay memory. It handles adding states to the replay memory and training the model (this is just standard deep Q learning).

## model.py

The model is formed of two convolutional layers, a flatten layer and two dense layers. This matches DeepMinds model (as far as I am aware).


## Testing and what I have learned.

### The beginning (very poor runtime)

When starting this project I just about copied an example from https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/. With a couple of modifications, replacing the environment and updating it to Tensorflow 2.0 it ran. The problem was it was predicting 311 days till it completed. This lead me to lower the replay memory size and episodes (I cannot remember what to now) which lowered the run time to about 3 days.

### Incorrect input 

the original input was an unmodified single frame for both the current state and next state (reward and action were fine though). This caused in increase in complexity and added in unnecessary data for training. to replace this I first updated it to grey scale which updated the training time to around a hour but later updated it further with a reduced frame size (to 84 x 84 pixels), cropped the score out of the frame and replaced the single frame for current state and next state to a combination of 4 frames each (the current state and next state became a Numpy array of shape 84x84x4). this did not affect the run time much but improved accuracy.

### Unequired layers in the model

For the model I started with a copy of the model I made for solving Mnist. This turned out to be a mistake as it included pooling and drop-out. Both of these are not required in deep Q learning as it adds noise on top of the noise already in the environment. Removing these layers decreased complexity and increased accuracy. This reduced training time down from 1 hour to around 10 minutes.

### Increasing episodes

Now training time was low enough I could increase the number of episodes (games played by the agent). With the current set-up I seemed to make some progress with around 1500 episodes but without increasing this to over 10000 it was very unlikely to get any descent results. For now I cannot try this due to limitations in hardware but will move to a machine with a GPU soon at which point I will hopefully be able to update this.

### Stopping episode on life loss

I am not sure if this works but I have tried to stop each episode on life lost instead of game over (after 5 lives). This in theory would mean losing a life would be of no benefit to the agent whereas if the game lasts for 5 lives then there way be some reward for letting the ball past the paddle. From what I have tried this just made the agent learn a specific move at a time before remaining still for all later moves. I think there may still be some work to improve this.


## Still to work out

During training what would often happen is the agent would learn if it only went right or only went left it would score between 4 and 11 points. This sometimes would get it stuck in a local minimum. I have tried clipping the rewards (which generally did help) and introduced stopping on life loss but am yet to truly solve this. I hope an increase of replay memory will be the solution but I will have to return in the future to try this.

## Conclusion

From the time of writing i have managed to get my agent to learn to hit the ball to score around 30 points when trained with 5 lives and 9 points when trained with 1 life. I feel the hardware I have is one of my big limiting factors as for instance the replay memory size is limited to 40000 as I only have 16GB of RAM. In addition to this until I have my desktop back I can only train on my CPU. I have definitely made progress but there is more work to be done.
