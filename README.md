# multi-agent-learning
Reinforcement Learning algorithm "multi-agent deep deterministic policy gradient" (MADDPG, [paper](https://arxiv.org/pdf/1706.02275.pdf)) is used to train openai's [multi-agent particle environments](https://github.com/openai/multiagent-particle-envs).

![multi-agent environment simple_tag](https://github.com/paxx13/multi-agent-learning/blob/origin/models/simple_tag.GIF "multi-agent environment simple_tag")

## Intallation
Dependencies with python 3.6
```
pip install gym
pip3 install torch
pip install tqdm
pip install tensorboardX
```
The multi-agent particle environments are included as a git submodule. After cloning the project you need to run 
```
git submodule init
git submodule update
```
and create a symbolic link to the multiagent folder
```
ln -s multiagent_envs\multiagent
```

## Usage
```
python main.py -h
usage: main.py [-h] [-t] [--env ENV] [--memory_size MEMORY_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           set to learn a policy
  --env ENV             name of the environment
  --memory_size MEMORY_SIZE
                        size of the replay buffer
```

## Troubleshooting
**Problem:** you encounter the following error `gym.error.Error: Cannot re-register id: MultiagentSimple-v0
**Solution:** you might need to change the ids to a different name in *\multiagent_envs\multiagent\__init__.py*

## Acknowledgements
The code is based on [multi-agent-rl](https://github.com/rohan-sawhney/multi-agent-rl) and refactored to make it more understandable. Furthermore tensorflow was replaced by a pytorch implementation which is based on [DDPG](https://github.com/samlanka/DDPG-PyTorch).
