# Perspective Taking in Deep Reinforcement Learning Agents
## Introduction
In this repository, you can find the code used for [Perspective taking in deep reinforcement learning agents (coming soon)](https://arxiv.org/).

We used APES environment to build a task that need perspective taking to be solved. The task consist of two agents *Subordinate* and *Dominant*. The task also include a food that the *Subordinate* should eat if it is not observed by the *Dominant*. The *Subordinate* was driven by a neural network trained using reinforcement learning.
## Installation 
The instruction for installation (using anaconda):
```
# Create an environment and install the required packages.
conda create -n APES_PT python=3 keras=2.0.8 tensorflow=1.10.0 jupyter scikit-learn scikit-images pandas matplotlib 
```
Some packages are not available in conda, so we need to install them using pip
```
# Activate the enviornment created
conda activate APES_PT
pip install scikit-video
```
## Training models or loading pretrained ones
### Training models: 
To train the models there are two files which use same parameters with one difference:
1. perspective_taking_normal.py: for experiments where allocentric vision use allocentric actions and egocentric vision use egocentric actions. 
2. perspective_taking_reversed.py: for experiments where allocentric vision use egocentric actions and viseversa.
Below is an example of using it.
```
python perspective_taking_normal.py 1 --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed 1337 --batch_size 16 --totalsteps 20000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 --Level 3 --Ego
```
### Load pre-trained models:
You can download the pre-trained models from [here](https://drive.google.com/file/d/1yQqOxCPAhApKMj0nZT1BT03hluNy-zDz/view?usp=sharing). You need to extract them into "output" folder in the same repository directory.

## Simulations
### Re-generate the simulations
To re-do the simulations, you need to have the training models (downloaded or re-trained). 
### Download simulations
To execute "Stage 1" and "Stage 2" jupyter notebooks you need to downloaded  [this](https://drive.google.com/file/d/1c_UZAYQdGAnRIkH21zfsMktGPtpzRCUj/view?usp=sharing) file and extract it into "NPZ" folder in same repository direction. 
