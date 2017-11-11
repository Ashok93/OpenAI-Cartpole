import gym
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = []
		self.learning_rate = 0.01
		self.exploration_rate = 1.0
		self.min_exploration_rate = 0.01
		self.exploration_rate_decay = 0.995
		self.discount_factor = 0.9
		self.model = self._build_model()


	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation = 'relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model

	def perform_action(self, state):
		self.exploration_rate *= self.exploration_rate_decay
		self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate)

		if np.random.rand() <= self.exploration_rate:
		 	return random.randrange(self.action_size)

		action_values = self.model.predict(state)
		return np.argmax(action_values[0])

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	def replay(self, batch_size):
		mini_batch = random.sample(self.memory, batch_size)
		
		for state, action, reward, next_state, done in mini_batch:
			actions = self.model.predict(state)
			
			actions[0][action] = reward
			if not done:
				actions[0][action] = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))

			self.model.fit(state, actions, epochs=1, verbose=0)


if __name__ == "__main__":

	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	batch_size = 32

	agent = DQNAgent(state_size, action_size)

	for episode in range(EPISODES):

		state = env.reset()
		state = np.reshape(state, [1, state_size])


		for t in range(500):

			action = agent.perform_action(state)
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			env.render()

			if done:
				print("episode {} at time {}".format(episode, t))
				if t >= 199:
					print('found optimal solution at episode {} at time {}'.format(episode, t))
					
				break

			if len(agent.memory) > batch_size:
				agent.replay(batch_size)