# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from cxcscmu/verl-agent-deepresearch (commit 9c311053).
# Only DeepResearchEnvironmentManager and the deepresearch branch of
# make_envs are kept (alfworld/sokoban/webshop/appworld/gym_cards managers
# and the SimpleMemory dependency are stripped).

from typing import List, Dict, Any
from functools import partial
import os
import json

from agent_system.environments.base import EnvironmentManagerBase, to_numpy


class DeepResearchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, dataset_name, is_train, is_evaluation):
        super().__init__(envs, projection_f, env_name)
        self.last_finished_idx = 0
        self.is_train = is_train
        self.is_evaluation = is_evaluation
        self._load_dataset(dataset_name)

    def reset(self) -> Dict[str, Any]:
        # assign the next unfinished question to the envs
        questions = []
        question_ids = []

        if self.envs.mode == "qa":
            ground_truths = []
        else:
            ground_truths = None

        for i in range(self.envs.env_num):  # env_num different questions for each env
            if self.last_finished_idx >= len(self.dataset):
                # reset cycle
                for item in self.dataset:
                    item['finished'] = False
                self.last_finished_idx = 0

            assert self.last_finished_idx < len(self.dataset), \
                f"last_finished_idx ({self.last_finished_idx}) >= dataset length ({len(self.dataset)})"

            if not self.dataset[self.last_finished_idx]['finished']:
                question = self.dataset[self.last_finished_idx]['question']
                question_id = self.dataset[self.last_finished_idx]['id']
                questions.append(question)
                question_ids.append(question_id)
                if self.envs.mode == "qa":  # qa with ground truth
                    ground_truths.append(self.dataset[self.last_finished_idx]['answer'])
                self.dataset[self.last_finished_idx]['finished'] = True
                self.last_finished_idx += 1

        print(f'question_ids: {question_ids}', flush=True)
        obs, infos = self.envs.reset(questions, question_ids, ground_truths)
        observations = {'text': obs, 'image': None, 'anchor': obs}
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids, original_responses = self.projection_f(text_actions)
        observations, rewards, dones, infos = self.envs.step(
            original_responses=original_responses, actions=actions)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        next_observations = {'text': observations, 'image': None, 'anchor': observations}

        return next_observations, rewards, dones, infos

    def _load_dataset(self, dataset_name):
        """Load the question dataset shipped with the env package."""
        dataset_dir = os.path.join(
            os.path.dirname(__file__),
            f"env_package/deepresearch/deepresearch/data/{dataset_name}")
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} does not exist")

        if self.is_train:
            data_path = os.path.join(dataset_dir, "train.json")
        elif self.is_evaluation:
            data_path = os.path.join(dataset_dir, "test.json")
        else:
            data_path = os.path.join(dataset_dir, "val.json")

        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item['finished'] = False


def make_envs(config):
    """Create environments — deepresearch only in this vendored copy."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    if "deepresearch" not in config.env.env_name.lower():
        raise ValueError(
            f"Environment {config.env.env_name} not supported — this vendored "
            f"copy only ships the deepresearch environment")

    from agent_system.environments.env_package.deepresearch import (
        build_deepresearch_envs, deepresearch_projection)
    max_steps = config.env.max_steps
    is_evaluation = config.env.is_evaluation
    use_explicit_thinking = config.env.use_explicit_thinking
    _envs = build_deepresearch_envs(
        dataset_name='train', seed=config.env.seed,
        env_num=config.data.train_batch_size, group_n=group_n,
        max_steps=max_steps, use_explicit_thinking=use_explicit_thinking)
    _val_envs = build_deepresearch_envs(
        dataset_name='val', seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size, group_n=1,
        max_steps=max_steps, use_explicit_thinking=use_explicit_thinking)

    projection_f = partial(deepresearch_projection)
    envs = DeepResearchEnvironmentManager(
        _envs, projection_f, config.env.env_name, config.env.dataset,
        is_train=True, is_evaluation=is_evaluation)
    val_envs = DeepResearchEnvironmentManager(
        _val_envs, projection_f, config.env.env_name, config.env.dataset,
        is_train=False, is_evaluation=is_evaluation)

    return envs, val_envs
