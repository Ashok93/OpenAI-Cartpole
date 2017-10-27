import gym
import numpy as np

def run_episode(env, params):
	observation = env.reset() #this has the env measurables such as angle, pos of cart etc. 4 values/params
	total_reward = 0
	
	for _ in range(200):
		env.render()
		action = 0 if np.matmul(params, observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		total_reward += reward

		if done:
			break

	return total_reward

def train():
	env = gym.make('CartPole-v0')
	parameters = np.random.rand(4)
	noise = 0.2
	counter = 0
	best_reward = 0

	for _ in range(1000):
		counter += 1 
		new_params = parameters + np.random.rand(4) * noise
		reward = run_episode(env, new_params)

		if reward > best_reward:
			best_reward = reward
			parameters = new_params

		print 'best reward for ', counter, ' iteration is ', best_reward
		if best_reward == 200:
			break

	return counter, parameters

no_of_tries, optimal_params = train()

print 'The total no. of tries taken to balance the pole by the program is ', no_of_tries
print 'The optimal set of params for controlling the pole is ', optimal_params