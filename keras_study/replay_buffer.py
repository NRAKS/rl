import numpy as np
import random
from collections import namedtuple
import keras

# Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
# blank_trans = Transition(0, np.zeros(84, 84), None, 0, False)


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer
        """

        self._storage = []
        self._maxisize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxisize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy = False))
            actions.append(np.array(action, copy = False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy = False))
            dones.append(done)
        return (np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones))

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


#セグメントツリー(優先順位づけのため)
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.sum_tree = [0] * (2 * size -1)
        self.data = [None]
        self.max = 1
    #インデックスツリーにvalueを伝える
    def _propagate(self, index, value):
        parent = (index -1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    #インデックスツリーにvalueを更新する
    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data
        self.update(self.index + self.size -1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    #全体木からvalueの位置を探す
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    #valueをツリーから探し、value、dataのインデックス、ツリーのインデックスを返す
    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

    

class ReplayMemory_Prioritized():
    def __init__(self, size, priority_weight):
        self._storage = []
        self._maxisize = size
        self._next_idx = 0
        self._priority_weight = [1]
        self._priority_weight_change = []
        self._transitions = SegmentTree(size)
        self._history = 1   #連続状態を処理する数
        self._multi_step = 1 #マルチステップ処理の数

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        self._transitions.append(Transition(self._next_idx, obs_t, action, reward, done), self._transitions.max)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1)  % self._maxisize
    


    #インデックス番号から各情報(行動、観測情報などを渡す)
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy = False))
            actions.append(np.array(action, copy = False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy = False))
            dones.append(done)
        return (np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones))

    def _get_sample_from_segment(self, segment, i):
        valid = False
        sample = random.uniform(i * segment, (i + 1) * segment)
        prob, idx, tree_idx = self._transitions.find(sample)
        if (self._transitions.index - idx) % self._maxisize > self._multi_step and (idx - self._transitions.index) % self._history and prob != 0:
            valid = True

    def sample(self, batch_size):
        p_total = self._transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]

        return self._encode_sample(batch)

    def update_priorities(self, idx, priorities):
        priorities.pow_