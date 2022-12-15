import tensorflow as tf
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


class NodeEnv(py_environment.PyEnvironment):
    def __init__(self, inputs, last_acc, original_path, state_path, classifier):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(inputs), name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(50,), dtype=np.int32, name='observation'
        )

        self.classifier = classifier
        self.classifier.save_weights(original_path)
        self.classifier.save_weights(state_path)
        self.original_path = original_path
        self.state_path = state_path

        self._original = inputs
        self._episode_ended = False
        self.original_acc = last_acc

        self._reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = deepcopy(self._original)
        self.classifier.load_weights(self.original_path)
        self.attacked_nodes = set()
        self.last_acc = self.original_acc
        return ts.restart(self._state)

    def _step(self, target_node):
        if (target_node in self.attacked_nodes) or (target_node > self._state.shape[0] - 50):
            return ts.termination(self._state, reward=-1)

        self.fake_node(target_node)

        if self.calc_perturb() >= 0.05 or len(self.attacked_nodes) >= 50:
            return ts.termination(self._state, reward=-1)

        rwd = self._get_reward()
        return ts.transition(self._state, reward=rwd, discount=1.0)

    def _get_reward(self):
        self.classifier.load_weights(self.state_path)

        history = self.classifier.fit(
            x=np.array([i for i in range(self._state[0])]),
            y=self._state[:, 0],
            epochs=20,
            batch_size=256,
            validation_split=0.15,
            verbose=0,
        )
        _, train_accuracy = self.classifier.evaluate(x=np.array([i for i in range(self._state[0])]), y=self._state[:, 0], verbose=0)
        res = self.last_acc - train_accuracy

        self.last_acc = train_accuracy
        self.classifier.save_weights(self.state_path)

        return res

    def calc_perturb(self):
        return 0.01

    def fake_node(self, target_node):
        fake_id = self._state.shape[0] - 50 + len(self.attacked_nodes) + 1
        self.attacked_nodes.add(target_node)

        label = np.random.randint(7)
        while label == self._state[target_node, 0]:
            label = np.random.randint(7)
        self._state[fake_id, 0] = label

        self._state[fake_id, 1:1434] = self._state[target_node, 1:1434]

        # modify edges
        self._state[fake_id, 1434+target_node] = 1
        self._state[target_node, 1434+fake_id] = 1

        for i in range(len(self._state)):
            if self._state[i, target_node+1434] == 1 and np.random.random() < 0.5:
                self._state[i, fake_id+1434] = 1
            elif self._state[target_node, i+1434] == 1 and np.random.random() < 0.5:
                self._state[fake_id, i+1434] = 1
