import numpy as np
import random
import gym


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Reach3dEnv(gym.Env):

    def __init__(self):


        self.ee_init_pos = np.array([7.50,-2.30,1.46])
        self.ee_pos = self.ee_init_pos
        self.reward_type="sparse"
        #self.goal = np.array([-2.49989452,5.72161969,-7.79728504])
        #self.goal = np.array([7.26933, -2.0233113, 1.533])
        self.goal = np.array([7.06933, -2.0233113, 1.933])

        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(3,))
        #print(self.get_joint_vel())
        self.action_high = 0.1
        self._max_episode_steps = 100
        self.step_counter = 0
        self.distance_threshold = 0.0001*1000
        self.action_type = "Continue"
        self.success_reward = 10.0


    def seed(self,seed):
        random.seed(seed)


    def get_observation(self):
        self.step_counter += 1
        done = False
        if self.step_counter > self._max_episode_steps:
            done = True
        reward = self.calculate_reward(self.ee_pos, self.goal)
        observation = {'observation': self.ee_pos,
                       "desired_goal": self.goal,
                       "achieved_goal": self.ee_pos,
                       "done": done,
                       "reward": reward
                       }
        return observation

    def reset_pos(self):
        #print(self.ee_init_pos)
        self.ee_pos = np.array([7.50,-2.30,1.46])
        #print(self.ee_pos)

    def reset(self):
        self.reset_pos()
        self.step_counter = 0
        #print(self.ee_pos)
        return self.get_observation()

    def inc_ee_pos(self, inspos) -> None:

        temp = self.ee_pos+inspos
        self.ee_pos = temp


    # step
    def calculate_reward(self, achieved_goal, goal,reward_type="sparse"):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # if self.reward_type == "sparse":
        #     return -(d > self.distance_threshold).astype(np.float32)
        # else:
        #     return -d
        if d < self.distance_threshold:
            return self.success_reward
        else:
            return -d * 10.0

    def step(self,action) -> None:
        self.inc_ee_pos(np.squeeze(action))

        return self.get_observation()
