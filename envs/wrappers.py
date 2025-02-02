import datetime
import gym
import numpy as np
import uuid
import pygame
import time

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class DaydreamerTimeLimit(TimeLimit):
    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            obs["is_last"] = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()

KEY_PAUSE = pygame.K_p
KEY_CONTINUE = pygame.K_c
KEY_RESET = pygame.K_r

class KBReset(gym.Wrapper):

  def __init__(self,env):
    super().__init__(env)
    pygame.init()
    self._screen = pygame.display.set_mode((800, 600))
    self._set_color((128, 128, 128))
    self._paused = False


  def step(self, action):
    pressed_last = self._get_pressed()
    #pressed_now = pygame.key.get_pressed()

    if self._paused:
        waiting = not (KEY_CONTINUE in pressed_last or KEY_RESET in pressed_last)
        hard = KEY_RESET in pressed_last
        while waiting:
            pressed = self._get_pressed()
            if KEY_CONTINUE in pressed or KEY_RESET in pressed:
                waiting = False
                hard = KEY_RESET in pressed
            self._set_color((255, 0, 0))  # Красный для паузы
            time.sleep(0.1)

        reset_action = action.get('reset', False) 
        if reset_action:
            #print(f"ACTION:{action.get('reset')}")

            # assert reset_action, action
                # Проверяем, что действие действительно сброс
            if not reset_action:
                raise ValueError(f"Invalid reset action: {action}")
        self._paused = False

        if hard:
            self.env.close() # Закрыди старое окружение , а потом запускаем новое 
            self.env = self.env.unwrapped  # Для пересоздания среду (окружение)
        obs, reward, done, info = self.env.step({**action, 'manual_resume': True})
        return obs, reward, done, info  # Требуется кортеж для gym.Env

    if KEY_PAUSE in pressed_last:
        self._set_color((255, 0, 0))  # Красный для паузы
        self._paused = True
        obs, reward, done, info = self.env.step({**action, 'manual_pause': True})
        return obs, reward, True, info  # происходит завершение эпизода

    self._set_color((128, 128, 128)) 
    obs, reward, done, info = self.env.step(action)

    sparse_rewards = [-1, 1, 10]
    if any(r - 0.01 < reward < r + 0.01 for r in sparse_rewards):  # Теперь reward из кортежа
        extra_info = f' | gripper_pos={info.get("gripper_pos", "N/A")}'  # Берем из info
        print(f'NON-ZERO REWARD: {reward}{extra_info}')

    return obs, reward, done, info  # Возвращаем корректный gym-кортеж

  def _get_pressed(self):
    pressed = []
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        pressed.append(event.key)
    return pressed

  def _set_color(self, color):
    self._screen.fill(color)
    pygame.display.flip()