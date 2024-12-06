# dreamerv3-torch
Pytorch implementation of [Daydreamer](https://github.com/danijar/daydreamer) only for quadrupedal robots. Based on Pytorch implementation of [DreamerV3](https://github.com/NM512/dreamerv3-torch).

In addition to adding Daydreamer features, the [DreamerV3](https://github.com/NM512/dreamerv3-torch) features are fully preserved (but not everything has been tested).

The project is not very well tested yet, but one can use it as a starting point at least.

## Instructions for DreamerV3

### Method 1: Manual (not tested)

Get dependencies with python 3.11:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```
To set up Atari or Minecraft environments, please check the scripts located in [env/setup_scripts](https://github.com/NM512/dreamerv3-torch/tree/main/envs/setup_scripts).

### Method 2: Docker (tested)

If one want to run training inside docker, for example for a1-robot, use:
```
docker run -it --rm --gpus all --net=host --env DISPLAY=$DISPLAY -v $PWD:/workspace dreamerv3 python3 dreamer.py \
   --configs dmc_vision --task dmc_walker_walk \
   --logdir "./logdir/dmc_walker_walk"
```
For more info please refer to the Dockerfile.

## Instructions for DayDreamer

### Method 1: Manual (not tested)

Get dependencies with python 3.11:
```
pip install -r requirements.txt
```
__Run training on a1__
In the first terminal, run:
```
python3 dreamer.py --configs a1 --task a1_sim --async_run learning --logdir ./logdir/a1
```
In the second terminal, run:
```
python3 dreamer.py --configs a1 --task a1_sim --async_run acting --logdir ./logdir/a1
```
__Monitor results__
In the third terminal, run:
```
tensorboard --logdir ./logdir
```

### Method 2: Docker (tested)

__Build docker image from Dockerfile__
From the root directory of the project, run:
```
docker build -t dreamerv3 .  
```
__Run training on a1__
In the first terminal, run:
```
docker run --name dreamerv3 -it --rm --gpus all --net=host --env DISPLAY=$DISPLAY -v $PWD:/workspace dreamerv3 python3 dreamer.py \
   --configs a1 --task a1_sim --async_run learning \
   --logdir "./logdir/a1"
```
In the second terminal, run:
```
docker exec -it dreamerv3 python3 dreamer.py \
   --configs a1 --task a1_sim --async_run acting \
   --logdir "./logdir/a1"
```
__Monitor results__
In the third terminal, run:
```
tensorboard --logdir ./logdir
```

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's DayDreamer tensorflow implementation: https://github.com/danijar/daydreamer
- NM512's dreamerv3-torch pytroch implementation of DreanerV3: https://github.com/danijar/daydreamer
