"""
python3
kerasの練習プログラム
actor-critic手法
"""
import os
import gym
from keras import backend as K
from keras import initializers as Init
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, RepeatVector
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Model

from replay_buffer import ReplayBuffer
from schedules import LinerSchedule
import numpy as np
import random
import time
import itertools

# actorクラス
class Actor():

    def __init__(self, env):
        self.env = env
        self.target = None

    def create_model(self, inpt, num_action, hidden_dims=[64, 64]):
        input_dim = Input(shape=(inpt, ))
        net = RepeatVector(inpt)(input_dim)
        net = Reshape([inpt, inpt, 1])(net)

        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)
        
        net = Flatten()(net)
        net = Dense(num_action)(net)
        
        self.model = Model(inputs=input_dim, outputs=net)
        self.model.compile('rmsprop', 'mse')

    def train(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch, y_batch)

    def predict(self, X_batch):
        return self.model.predict_on_batch(X_batch)

    def act(self, obs):
        obs = obs.reshape(-1, env.observation_space.shape[0])
        Q = self.model.predict_on_batch(obs)
        return np.argmax(Q, 1)[0]


# criticクラス
class Critic():
    def __init__(self, env):
        self.env = env
        self.target = None

    def create_model(self, inpt, num_action, hidden_dims=[64, 64]):
        
        input_dim = Input(shape=(inpt, ))
        net = RepeatVector(inpt)(input_dim)
        net = Reshape([inpt, inpt, 1])(net)

        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)
        
        net = Flatten()(net)
        net = Dense(num_action)(net)
        
        self.model = Model(inputs=input_dim, outputs=net)
        self.model.compile('rmsprop', 'mse')

    def update_target(self):
        config = self.model.get_config()
        self.target = Model.from_config(config)
        weights = self.model.get_weights()
        self.target.set_weights(weights)

    def train(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch, y_batch)

    def predict(self, X_batch):
        return self.model.predict_on_batch(X_batch)
    
    def target_predict(self, X_batch):
        return self.target.predict_on_batch(X_batch)


def create_batch(agent, replay_buffer, batch_size, discount_rate):
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer

    X_batch = np.vstack(obses_t)
    y_batch = agent.predict(X_batch)

    y_batch[np.arange(batch_size), actions] = (rewards
                                               + discount_rate
                                               * np.max(agent.target_predict(np.vstack(obses_tp1)), 1)
                                               * (1 - dones))

    print("TD_error:{}".format(y_batch))

    return X_batch, y_batch


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    actor = Actor(env)
    critic = Critic(env)
    actor.create_model(inpt=env.observation_space.shape[0], num_action=env.action_space.n)

    critic.create_model(inpt=env.observation_space.shape[0], num_action=env.action_space.n)

    replay_buffer = ReplayBuffer(50000)
    exploration = LinerSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

    # 初期化
    discount_rate = 0.99
    episode_rewards = [0.0]
    obs = env.reset()
    critic.update_target()

    for t in itertools.count():
        
        action = actor.act(obs)

        new_obs, rew, done, _ = env.step(action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        episode_rewards[-1] += rew

        if done:
            obs = env.reset()
            print("t:{}, episode_rewards:{}, eps:{}"
                  .format(t, episode_rewards[-1], exploration.value(t)))
            episode_rewards.append(0)

        is_solved = t > 100 and np.mean(episode_rewards[-51:-1]) >= 170

        if is_solved:
            env.render()
        else:
            if t > 1000:
                # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                X_batch, y_batch = create_batch(critic, replay_buffer.sample(32), 32, discount_rate)
                critic.train(X_batch, y_batch)
                actor.train(X_batch, y_batch)
            if t % 1000 == 0:
                critic.update_target()