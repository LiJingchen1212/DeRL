import gym
import random
import numpy as np
from gym.utils import seeding

class Watering_cart(gym.Env):
    def __init__(self):
        self.cart_x = 0.0
        self.cart_y = 0.0
        #self.cart_vx = 0.0
        #self.cart_vy = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.radius = 1.0
        self.target_value = 1.0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,))
        self.step_counter = 0
        self._max_episode_steps = 50

    #def seed(self, seed=None):
        #self.np_random, seed = seeding.np_random(seed)
        #return [seed]
        
    def seed(self, seed):
        random.seed(seed)

    def reset(self):
        self.cart_x = 0.0
        self.cart_y = 0.0
        self.target_x = random.uniform(-9.0, 9.0)
        self.target_y = random.uniform(-9.0, 9.0)

        self.step_counter = 0
        state = np.array((self.cart_x, self.cart_y, self.target_x, self.target_y))
        return state

    def step(self, action):
        action = np.squeeze(action)
        self.cart_x = self.cart_x + action[0]
        self.cart_y = self.cart_y + action[1]
        reward = 0.0
        flag = False
        if self.cart_x > 10.0:
            self.cart_x = 10.0
            flag = True
        if self.cart_x < -10.0:
            self.cart_x = -10.0
            flag = True
        if self.cart_y > 10.0:
            self.cart_y = 10.0
            flag = True
        if self.cart_y < -10.0:
            self.cart_y = -10.0
            flag = True
        if flag:
            reward -= 1.0

        distance = (self.cart_x - self.target_x)*(self.cart_x - self.target_x)+(self.cart_y - self.target_y)*(self.cart_y - self.target_y)
        if distance > self.radius*self.radius:
            reward -= abs(action[2])
        else:
            reward += 1 - abs(self.target_value - action[2])

        self.step_counter += 1

        done = False
        if self.step_counter >= self._max_episode_steps:
            done = True

        state = (self.cart_x, self.cart_y, self.target_x, self.target_y)
        info = {}
        return np.array(state), reward, done, info

