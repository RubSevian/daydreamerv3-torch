import numpy as np
import gym


class LeggedRobot():

    def __init__(
        self,
        task: str,
        robot_type: str,
        repeat: int = 1,
        length: int = 1000,
        resets: bool = True,
        enable_rendering: bool | None = True
    ):
        assert robot_type in ("A1", "Go1","Aliengo"), "Incorrect robot type"
        assert task in ("sim", "real"), task

        # don't move the import from this place! It works only like this
        from motion_imitation.envs import env_builder

        self._env = env_builder.build_env(
            enable_rendering=enable_rendering,
            num_action_repeat=repeat,
            use_real_robot=bool(task == "real"),
            robot_type = robot_type
        )

    @property
    def obs_space(self):
        return gym.spaces.Dict({
            **self._env.obs_space,
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        })

    @property
    def act_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
    
    def step(self, action):
        obs = self._env.step(action)
        obs["image"] = self._gymenv.render("rgb_array")
        assert obs["image"].shape == (64, 64, 3), obs["image"].shape
        assert obs["image"].dtype == np.uint8, obs["image"].dtype
        return obs
