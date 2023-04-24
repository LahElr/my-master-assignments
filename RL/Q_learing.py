import logging
import random
# from collections import defaultdict
from typing import Iterable, Tuple, Union
import time
import matplotlib
from matplotlib import pyplot
import os

import numpy
from tqdm import tqdm, trange

from environment import CliffBoxPushingBase

StateType = Tuple[Tuple[int, int], Tuple[int, int]]
exp_title = "box_goal_dist_on_sarsa"
agent_mode = "q"  # "q" or "sarsa"
box_goal_dist = True

if not os.path.exists(f"./{exp_title}"):
    os.makedirs(f"./{exp_title}")

# format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logging.basicConfig(filename=f"{exp_title}/log.log", filemode="a")

logger = logging.getLogger("rl")
logger.setLevel(logging.DEBUG)
logger.info("\n"+"-"*7+"New log started"+"-"*7+"\n")


def is_iterable(x: object) -> bool:
    '''This function aims at telling if the object is iterable by telling if it has `__getitem__` function
    '''
    # return hasattr(x, "__getitem__")
    return hasattr(x, "__iter__")


def flatten_list(x: Iterable) -> Iterable:
    '''This function can recursively flatten an iterable object
    '''
    if not is_iterable(x):
        raise TypeError()
    ret = []
    for item in x:
        if is_iterable(item):
            ret.extend(flatten_list(item))
        else:
            ret.append(item)
    return ret


class QTable(numpy.ndarray):
    '''
    This class is subclass of `numpy.ndarray`, aims at faster and more convenient indexing over Q table
    When using, manually init an `ndarray` and use view to convert to `QTable`

    eg:
    ```
    x = numpy.zeros((2,3,3),numpy.float16)
    x = x.view(QTable)
    ```
    '''

    def __getitem__(self, key):
        '''use either `[state]` or `[state, action]` to index
        '''
        key = tuple(flatten_list(key))
        return super(QTable, self).__getitem__(key)

    def __setitem__(self, key, value):
        '''use either `[state]` or `[state, action]` to index
        '''
        key = tuple(flatten_list(key))
        return super(QTable, self).__setitem__(key, value)


class Agent:
    def __init__(self):
        # super-params
        self.discount_factor = 0.99  # gamma
        self.alpha = 0.5
        self.epsilon = 0.01

        self.action_space = [1, 2, 3, 4]
        # self.action_space = bidict(enumerate(self.action_space)) # action_index - action
        # self.V = []
        # self.Q = defaultdict(lambda: numpy.zeros(len(self.action_space)))
        self.Q = numpy.zeros((env.world_height, env.world_width, env.world_height,
                             env.world_width, len(self.action_space)), numpy.float16)
        self.Q = self.Q.view(QTable)
        # self.Q = QTable((env.world_height, env.world_width, env.world_height, env.world_width), len(self.action_space))

    def take_action(self, state: StateType) -> int:
        '''returns action index
        '''
        # eps-greedy
        if random.random() < self.epsilon:
            # action = random.choice(self.action_space)
            action = random.randint(0, len(self.action_space)-1)
        else:
            # action = self.action_space[numpy.argmax(self.Q[state])]
            action = int(numpy.argmax(self.Q[state]))
        return action

    def train_step(self, state: StateType, action: int, next_state: StateType, reward: int) -> None:
        raise NotImplementedError

    '''
    env.step([1,2,...])，虽然参数是一个列表，但只能执行其中的第一步（不过记录会全部写上），返回reward，是否结束，和一个注释中称为info的东西，它永远都是空的
    env.get_obs返回长度12的数组，前9个是周围环境，-5是空气，-1是自己，-2是箱子,-4是cliff，推断-3是目标；后面2个是自己的坐标，最后一个是距离箱子的距离
    env._states并不是文档中说的states，是整个地图的状态
    '''


class QAgent(Agent):
    def train_step(self, state: StateType, action: int, next_state: StateType, reward: int) -> None:
        '''action index
        '''
        self.Q[state, action] = self.Q[state, action]+self.alpha * \
            (reward+self.discount_factor *
             numpy.max(self.Q[next_state])-self.Q[state, action])


class SarsaAgent(Agent):
    def train_step(self, state: StateType, action: int, next_state: StateType, reward: int, action_prime: int) -> None:
        '''action index
        '''
        self.Q[state, action] = self.Q[state, action]+self.alpha * \
            (reward+self.discount_factor *
             self.Q[next_state, action_prime]-self.Q[state, action])


if __name__ == '__main__':
    # parameters
    num_epochs = 2500
    best_save_position = f"./{exp_title}/best.npy"

    # set up
    env = CliffBoxPushingBase(reward_offcliff=-10000,
                              reward_box_goal_distance=box_goal_dist)
    teminated = False
    rewards = []
    actions = []
    reward_records = []
    best_reward_records = [[], []]
    time_step = 0
    process_bar = trange(num_epochs)
    sum_reward = -99999
    max_sum_reward = -99999

    # start training
    if agent_mode == "q":
        agent = QAgent()
        start_time = time.time()
        logger.info(f"start training at time {time.ctime(start_time)}")
        for i in process_bar:
            process_bar.set_description(
                f"Training epoch {i}/{num_epochs}, rewards: {sum_reward}")
            env.reset()

            while not teminated:
                # choose A from S
                state = env.get_state()  # StateType
                action_i = agent.take_action(state)  # A_i
                actions.append(action_i)
                action = agent.action_space[action_i]  # A

                # take action A and get R and S'
                reward, teminated, _ = env.step([action])  # R
                next_state = env.get_state()  # S'
                rewards.append(reward)

                # train_step
                agent.train_step(state, action_i, next_state, reward)
                time_step += 1

            sum_reward = sum(rewards)
            reward_records.append(sum_reward)

            if sum_reward >= max_sum_reward:
                numpy.save(best_save_position, agent.Q)
                max_sum_reward = sum_reward
                logger.info(
                    f"Best reward at {i} is {max_sum_reward}, action sequence is {actions}")
                best_reward_records[1].append(max_sum_reward)
                best_reward_records[0].append(i)
            # print(f'print the historical actions: {env.episode_actions}')
            teminated = False
            rewards = []
            actions = []

    elif agent_mode == "sarsa":
        agent = SarsaAgent()
        start_time = time.time()
        logger.info(f"start training at time {time.ctime(start_time)}")
        for i in process_bar:
            raise NotImplementedError(f"SARSA episode training not implented.")

    end_time = time.time()
    logger.info(f"end training at time {time.ctime(end_time)}")
    end_info = f"training completed, best reward is {max_sum_reward}, overall time used is {end_time-start_time} seconds."
    print(end_info)
    logger.info(end_info)

    fig, ax = pyplot.subplots(figsize=(15., 6.))
    ax.plot(range(0, num_epochs), reward_records,
            label="reward", color="lightskyblue", linewidth=0.5)
    ax.set_title("learning progress")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.plot(best_reward_records[0], best_reward_records[1],
            "o-", label="best_reward_ever", color="r")
    ax.legend()
    ax.set_yscale("symlog")
    best_value = -388 if box_goal_dist else -90
    ax.set_yticks([best_value, -500, -1000, -2000, -5000, -10000])
    # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.set_yticklabels(["$%.1f$" % y for y in yticks]
    pyplot.savefig(f"./{exp_title}/learning progress.png",
                   dpi=300, bbox_inches="tight")
