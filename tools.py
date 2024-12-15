import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random

import numpy as np
from parallel import Parallel, Damy
from typing import OrderedDict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step, action_repeat, is_daydreamer):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._action_repeat = action_repeat
        self._is_daydreamer = is_daydreamer
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)
    
    def increment_step(self):
        # for daydreamer (aka for the real robot),
        # the action_repeat value should't be considered
        # for the logger step, as it is used only
        # to interpolate between two joint positions
        if self._is_daydreamer:
            self.step += 1
        else:
            self.step += 1 * self._action_repeat

    @property
    def is_daydreamer(self):
        return self._is_daydreamer

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


class Runner:
    """
    A class used to manage the learning process

    ...

    Attributes
    ----------
    envs : list[Parallel] | list[Damy]
        gym envs wrapped in Parallel or Damy
    cache : OrderedDict
        replay buffer returned by load_episodes function
        looks like {env.id: {"MotorAngle": [...], "IMU": [...], ...}}
    directory : pathlib.Path
        directory from config.traindir
        (usually config.logdir/"train_eps")
    logger : Logger
    max_dataset_size : None | int
        maximum number of items in the lists inside the cache
    """

    def __init__(
        self,
        envs: list[Parallel] | list[Damy],
        cache: OrderedDict,
        directory: pathlib.Path,
        logger: Logger,
        max_dataset_size: None | int = None,
    ) -> None:
        self._envs = envs
        self._cache = cache
        self._directory = directory
        self._logger = logger
        self._state = self._initialize_state()
        self._max_dataset_size = max_dataset_size

    def _initialize_state(self):
        result = {}
        result["done"] = np.ones(len(self._envs), bool)
        result["length"] = np.zeros(len(self._envs), np.int32)  # maybe useless
        result["obs"] = [None] * len(self._envs)
        result["agent_state"] = None
        result["reward"] = [0] * len(self._envs)  # maybe useless
        return result

    def _reset_envs(self):
        indices = [index for index, done in enumerate(self._state["done"]) if done]
        results = [self._envs[i].reset() for i in indices]
        results = [r() for r in results]
        return_dict = {}
        for index, result in zip(indices, results):
            t = result.copy()
            t = {k: convert(v) for k, v in t.items()}
            # action will be added to transition in add_to_cache
            t["reward"] = 0.0
            t["discount"] = 1.0
            # initial state should be added to cache
            return_dict[self._envs[index].id] = t
            # replace obs with done by initial state
            self._state["obs"][index] = result
        return return_dict

    def _step_agent(self, agent, prefill_run: bool, training: bool = True):
        obs = {k: np.stack([obs[k] for obs in self._state["obs"]]) for k in self._state["obs"][0] if "log_" not in k}

        # agents other than Dreamer may not have the training arg
        # a different agent is used during the prefill run
        if not prefill_run:
            action, self._state["agent_state"] = agent(obs, self._state["done"], self._state["agent_state"], training=training)
        else:
            action, self._state["agent_state"] = agent(obs, self._state["done"], self._state["agent_state"])

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(self._envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(self._envs)
        return action

    def _prepare_replay(self, action, results):
        return_dict = {}
        for a, result, env in zip(action, results, self._envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            return_dict[env.id] = transition
        return return_dict

    def _pre_log(self, i) -> float:
        length = len(self._cache[self._envs[i].id]["reward"]) - 1
        score = float(np.array(self._cache[self._envs[i].id]["reward"]).sum())
        # record logs given from environments
        for key in list(self._cache[self._envs[i].id].keys()):
            if "log_" in key:
                self._logger.scalar(
                    key, float(np.array(self._cache[self._envs[i].id][key]).sum())
                )
                # log items won't be used later
                self._cache[self._envs[i].id].pop(key)
        return length, score

    def _log_train_info(self, i):
        length, score = self._pre_log(i)
        step_in_dataset = erase_over_episodes(self._cache, self._max_dataset_size)
        self._logger.scalar(f"dataset_size", step_in_dataset)
        self._logger.scalar(f"train_return", score)
        self._logger.scalar(f"train_length", length)
        self._logger.scalar(f"train_episodes", len(self._cache))
        self._logger.write(step=self._logger.step)

    def _step_envs(self, action):
        # step envs
        results = [e.step(a) for e, a in zip(self._envs, action)]
        results = [r() for r in results]
        self._state["obs"], self._state["reward"], self._state["done"] = zip(*[p[:3] for p in results])

        # TODO:
        # check for importance of converting to list
        self._state["obs"] = list(self._state["obs"])
        self._state["reward"] = list(self._state["reward"])
        self._state["done"] = np.stack(self._state["done"])
        self._state["length"] += 1
        return results

    def run(self, agent, steps=0, episodes=0, prefill_run=False):
        """Interacts with the gym env and collects replay

        Parameters
        ----------
        agent : Dreamer or random_agent
        steps : int
            The number of training steps to be performed
            (It doesn't take action_reapeat into account like a normal Dreamer does)
        episodes : int
            The number of training episodes to be performed
        prefill_run : bool
            Usually random_agent is used to execute prefill_run
        """

        step = 0
        episode = 0
        while (steps and step < steps) or (episodes and episode < episodes):
            # reset envs before agent step if necessary
            if self._state["done"].any():
                transition = self._reset_envs()
                for key, value in transition.items():
                    add_to_cache(self._cache, key, value)

            # main training part
            action = self._step_agent(agent, prefill_run)
            results = self._step_envs(action)
            t = self._prepare_replay(action, results)
            for key, value in t.items():
                add_to_cache(self._cache, key, value)
            self._logger.increment_step()

            if self._state["done"].any():
                indices = [index for index, done in enumerate(self._state["done"]) if done]
                # logging for done episode
                for i in indices:
                    save_episodes(self._directory, {self._envs[i].id: self._cache[self._envs[i].id]})
                    self._log_train_info(i)
            step += len(self._envs)
            episode += int(self._state["done"].sum())


class EvalRunner(Runner):
    """
    A class used to manage the evaluation process

    ...

    Attributes
    ----------
    envs : list[Parallel] | list[Damy]
        gym envs wrapped in Parallel or Damy
    cache : OrderedDict
        replay buffer returned by load_episodes function
        looks like {env.id: {"MotorAngle": [...], "IMU": [...], ...}}
    directory : pathlib.Path
        directory from config.evaldir
        (usually config.logdir/"eval_eps")
    logger : Logger
    max_dataset_size : None | int
        maximum number of items in the lists inside the cache
    """

    def __init__(self, envs, cache, directory, logger, max_dataset_size = None):
        super().__init__(envs, cache, directory, logger, max_dataset_size)
        self._eval_lengths = []
        self._eval_scores = []
        self._eval_done = False

    def _log_eval_info(self, i, episodes):
        length, score = self._pre_log(i)
        video = self._cache[self._envs[i].id]["image"]

        # start counting scores for evaluation
        self._eval_scores.append(score)
        self._eval_lengths.append(length)

        mean_score = sum(self._eval_scores) / len(self._eval_scores)
        mean_length = sum(self._eval_lengths) / len(self._eval_lengths)
        self._logger.video(f"eval_policy", np.array(video)[None])

        if len(self._eval_scores) >= episodes and not self._eval_done:
            self._logger.scalar(f"eval_return", mean_score)
            self._logger.scalar(f"eval_length", mean_length)
            self._logger.scalar(f"eval_episodes", len(self._eval_scores))
            self._logger.write(step=self._logger.step)
            self._eval_done = True

    def _reset_eval_state(self):
        super()._initialize_state()
        self._eval_lengths = []
        self._eval_scores = []
        self._eval_done = False

    def eval_run(self, agent, steps=0, episodes=0):
        step = 0
        episode = 0
        while (steps and step < steps) or (episodes and episode < episodes):
            # reset envs before agent step if necessary
            if self._state["done"].any():
                transition = self._reset_envs()
                for key, value in transition.items():
                    add_to_cache(self._cache, key, value)

            # main part
            action = self._step_agent(agent, prefill_run=False, training=False)
            results = self._step_envs(action)
            t = self._prepare_replay(action, results)
            for key, value in t.items():
                add_to_cache(self._cache, key, value)

            if self._state["done"].any():
                indices = [index for index, done in enumerate(self._state["done"]) if done]
                # logging for done episode
                for i in indices:
                    save_episodes(self._directory, {self._envs[i].id: self._cache[self._envs[i].id]})
                    self._log_eval_info(i, episodes)
            step += len(self._envs)
            episode += int(self._state["done"].sum())

        # keep only last item for saving memory.
        # this cache is used for video_pred later
        while len(self._cache) > 1:
            # FIFO
            self._cache.popitem(last=False)
        # reset states after each evaluation run
        self._reset_eval_state()


class AsyncReplayCollector(Runner):
    """
    A class used to interact with envs and collect replay,
    but not training the agent

    ...

    Attributes
    ----------
    envs : list[Parallel] | list[Damy]
        gym envs wrapped in Parallel or Damy
    cache : FixedLength
        replay buffer returned by dreamer.make_replay() function
        looks like {env.id: {"MotorAngle": [...], "IMU": [...], ...}}
    directory : pathlib.Path
        directory from config.traindir
        (usually config.logdir/"train_eps")
    sync_directory: pathlib.Path
        directory from config.syncdir
        (usually config.logdir/"sync_weights")
    logger : Logger
    sync_every : int
        once in sync_every steps to load weights from the trained agent
        from AsyncLearner
    max_dataset_size : None
        is not used here
    """

    def __init__(self, envs, cache, directory, sync_directory, logger, sync_every, max_dataset_size = None):
        super().__init__(envs, cache, directory, logger, max_dataset_size)
        self._sync_every = sync_every
        self._sync_directory = sync_directory
        assert self._logger.is_daydreamer

    def _log_actor_info(self, episode: dict[str, list]):
        length = len(episode["reward"])
        score = float(np.array(episode["reward"]).sum())
        mean_reward = score / length
        self._logger.scalar(f"train_episodes", self._cache.stats["replay_episodes"])
        self._logger.scalar(f"train_return", score)
        self._logger.scalar(f"train_length", length)
        self._logger.scalar(f"mean_reward", mean_reward)
        self._logger.write(step=self._logger.step)

    def run(
        self, agent, steps: int = 0, episodes: int = 0, prefill_run: bool = False
    ):
        """Interacts with the gym env (or real robot) and collects replay

        Parameters
        ----------
        agent : Dreamer or random_agent
        steps : int
            The number of training steps to be performed
            (It doesn't take action_reapeat into account like a normal Dreamer does)
        episodes : int
            The number of training episodes to be performed
        prefill_run : bool
            Usually random_agent is used to execute prefill_run
        """

        step = 0
        episode = 0
        episodes_cache = collections.defaultdict(lambda: collections.defaultdict(list))
        while (steps and step < steps) or (episodes and episode < episodes):
            # reset envs before agent step if necessary
            if self._state["done"].any():
                self._reset_envs()

            action = self._step_agent(agent, prefill_run, training=False)
            results = self._step_envs(action)
            transition = self._prepare_replay(action, results)
            # transition is {env.id: {"MotorAngle": [...], "IMU": [...], ...}}
            for key, value in transition.items():
                self._cache.add(value, key)
                # store the transition for logging after done
                [episodes_cache[key][k].append(v) for k, v in value.items()]
            self._logger.increment_step()

            if self._state["done"].any():
                indices = [index for index, done in enumerate(self._state["done"]) if done]
                # logging for done episode
                for i in indices:
                    self._log_actor_info(episodes_cache[self._envs[i].id])
                    # clear the episodes_cache for the env that received the done flag
                    episodes_cache[self._envs[i].id].clear()

            if not prefill_run and step % self._sync_every == 0:
                while not (self._sync_directory / "latest_sync.pt").exists():
                    print("[AsyncReplayCollector] Waiting for agent checkpoint to be created")
                    time.sleep(10)
                print(f"[{self._logger.step}][AsyncReplayCollector] Syncing")
                try:  # to avoid simultaneously saving and loading the state dict
                    checkpoint = torch.load(self._sync_directory / "latest_sync.pt", weights_only=True)
                    agent.load_state_dict(checkpoint["agent_state_dict"])
                except Exception as e:
                    print(f"[{self._logger.step}][AsyncReplayCollector] {e}")
            
            step += len(self._envs)
            episode += int(self._state["done"].sum())


class AsyncLearner:
    """
    A class used to train the agent by sampling replay from the cache
    ...

    Attributes
    ----------
    cache : FixedLength
        eplay buffer returned by dreamer.make_replay() function
        looks like {env.id: {"MotorAngle": [...], "IMU": [...], ...}}
    directory : pathlib.Path
        directory from config.logdir
    sync_directory: pathlib.Path
        directory from config.syncdir
        (usually config.logdir/"sync_weights")
    logger : Logger
    sync_every : int
        once in sync_every steps to save agent weights
    save_every : int
        once in sync_every steps to save agent parameters
        this is necessary to restore training from the checkpoint
    """

    def __init__(self, cache, directory, sync_directory, logger: Logger, sync_every, save_every):
        self._cache = cache
        self._directory = directory
        self._logger: Logger = logger
        self._sync_every = sync_every
        self._sync_directory = sync_directory
        self._save_every = save_every
        assert self._logger.is_daydreamer

    def run(self, agent, prefill_steps, steps):
        """Trains the agent by sampling replay from the cache

        Parameters
        ----------
        agent : Dreamer
        prefill_steps : int
            it just waits for the cache to prefill the prefill_steps amount of data
            to avoid overfitting on a small amount of initial data
        steps : int
            The number of training steps to be performed
        """

        # Wait for prefill data from at least one actor to avoid overfitting to only
        # small amount of data that is read first.
        while len(self._cache) < prefill_steps:
            print(
                'Waiting for train data prefill '
                f'({len(self._cache)}/{prefill_steps})...')
            time.sleep(30)

        while self._logger.step < steps:
            agent.train()
            self._logger.increment_step()
            if self._logger.step % self._sync_every == 0:
                print(f"[{self._logger.step}][AsyncLearner] Syncing")
                torch.save(
                    {"agent_state_dict": agent.state_dict()},
                    self._sync_directory / "latest_sync.pt",
                )
            # saving checkpoints
            if self._logger.step % self._save_every == 0:
                items_to_save = {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": recursively_collect_optim_state_dict(agent),
                }
                torch.save(items_to_save, self._directory / "latest.pt")


class DEPRECATEDAsyncLearner(Runner):
    """
    To delete
    """
    def __init__(self, envs, cache, directory, sync_directory, logger, sync_every, max_dataset_size = None):
        super().__init__(envs, cache, directory, logger, max_dataset_size)
        self._sync_every = sync_every
        self._sync_directory = sync_directory
        assert self._logger.is_daydreamer

    def _initialize_state(self):
        result = {}
        result["agent_state"] = None
        return result

    def _async_step_agent(self, agent, observation, done) -> None:
        obs = {k: np.stack([obs[k] for obs in observation]) for k in observation[0] if "log_" not in k}
        _, self._state["agent_state"] = agent(obs, done, self._state["agent_state"], training=True)

    def run_learning(self, agent, prefill_steps, steps=0):
        """
        TODO: Add description
        """

        # Wait for prefill data from at least one actor to avoid overfitting to only
        # small amount of data that is read first.
        while len(self._cache) < prefill_steps:
            print(
                'Waiting for train data prefill '
                f'({len(self._cache)}/{prefill_steps})...')
            time.sleep(30)

        while self._logger.step < steps:
            agent.train()
            self._logger.increment_step()
            if self._logger.step % self._sync_every == 0:
                print(f"[{self._logger.step}][AsyncLearner] Syncing")
                torch.save(
                    {"agent_state_dict": agent.state_dict()},
                    self._sync_directory / "latest_sync.pt",
                )


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2)
            - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
