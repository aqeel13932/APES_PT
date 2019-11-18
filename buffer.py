# SOURCE : https://raw.githubusercontent.com/tambetm/gymexperiments/master/buffer.py
import numpy as np
import random

class Buffer:
    def __init__(self, size, observation_shape, action_shape, observation_dtype=np.float, action_dtype=np.integer, reward_dtype=np.float):
        self.size = size
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.preobs = np.empty((self.size,) + observation_shape, dtype=observation_dtype)
        self.actions = np.empty((self.size,) + action_shape, dtype=action_dtype)
        self.rewards = np.empty(self.size, dtype=reward_dtype)
        self.postobs = np.empty((self.size,) + observation_shape, dtype=observation_dtype)
        self.terminals = np.empty(self.size, dtype=np.bool)

        self.count = 0
        self.current = 0

    def add(self, preobs, action, reward, postobs, terminal):
        assert preobs.shape == self.observation_shape
        assert action.shape == self.action_shape
        assert postobs.shape == self.observation_shape
        self.preobs[self.current] = preobs
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.postobs[self.current] = postobs
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def addBatch(self, preobs, actions, rewards, postobs, terminals):
        for preob, action, reward, postob, terminal in zip(preobs, actions, rewards, postobs, terminals):
            self.add(preob, action, reward, postob, terminal)

    def sample(self, batch_size):
        assert self.count > 0
        indexes = np.random.choice(self.count, size=batch_size)
        return self.preobs[indexes], self.actions[indexes], self.rewards[indexes], self.postobs[indexes], self.terminals[indexes]

class BufferCNN:
    def __init__(self, size, observation_shape,observation2_shape, action_shape, observation_dtype=np.float, action_dtype=np.integer, reward_dtype=np.float):
        self.size = size
        self.observation_shape = observation_shape
        self.observation2_shape = observation2_shape
        self.action_shape = action_shape

        self.preobs = np.empty((self.size,) + observation_shape, dtype=observation_dtype)
        self.preobs2 = np.empty((self.size,) + observation2_shape, dtype=observation_dtype)
        self.actions = np.empty((self.size,) + action_shape, dtype=action_dtype)
        self.rewards = np.empty(self.size, dtype=reward_dtype)
        self.postobs = np.empty((self.size,) + observation_shape, dtype=observation_dtype)
        self.postobs2 = np.empty((self.size,) + observation2_shape, dtype=observation_dtype)
        self.terminals = np.empty(self.size, dtype=np.bool)

        self.count = 0
        self.current = 0

    def add(self, preobs,preobs2,action, reward, postobs, postobs2,terminal):
        assert preobs.shape == self.observation_shape
        assert preobs2.shape == self.observation2_shape
        assert action.shape == self.action_shape
        assert postobs.shape == self.observation_shape
        assert postobs2.shape == self.observation2_shape
        self.preobs[self.current] = preobs
        self.preobs2[self.current] = preobs2
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.postobs[self.current] = postobs
        self.postobs2[self.current] = postobs2
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def addBatch(self, preobs,preobs2, actions, rewards, postobs,postobs2, terminals):
        for preob, preob2, action, reward, postob,postob2, terminal in zip(preobs, preobs2, actions, rewards, postobs,postobs2, terminals):
            self.add(preob, preob2, action, reward, postob,postob2, terminal)

    def sample(self, batch_size):
        assert self.count > 0
        indexes = np.random.choice(self.count, size=batch_size)
        return self.preobs[indexes], self.preobs2[indexes], self.actions[indexes], self.rewards[indexes], self.postobs[indexes],self.postobs2[indexes], self.terminals[indexes]

    def  allOrdered(self):
        indexes = range(self.current)
        return self.preobs[indexes],self.preobs2[indexes], self.actions[indexes], self.rewards[indexes], self.postobs[indexes],  self.postobs2[indexes], self.terminals[indexes]

    def empty(self):
        self.count = 0
        self.current = 0

class BufferLSTM:
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size):
        sampled_episodes = random.sample(self.buffer,batch_size)  #Changed from random.sample
        out = []
        for i in range(len(sampled_episodes[0])):
            value = np.stack([sampled_episodes[j][i] for j in range(batch_size)],axis=0)
            out.append(value)
        return out

