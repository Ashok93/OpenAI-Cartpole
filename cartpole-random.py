import numpy as np
import gym

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



def train_model():
	env = gym.make('CartPole-v0')
	best_reward = 0
	best_param = None
	counter = 0

	for _ in range(10000):
		counter += 1
		parameters = np.random.rand(4)
		reward = run_episode(env, parameters) #compute reward for that particular set of random params

		if reward > best_reward:
			best_reward = reward
			best_param = parameters

		print "best reward at ", counter, "iteration is", best_reward		

		if reward == 200:
			break

	return counter, best_param

no_of_tries, params = train_model()

print 'Total tries made by the program to find optimum parameter for balancing is ', no_of_tries
print 'The parameters for successful control are ', params