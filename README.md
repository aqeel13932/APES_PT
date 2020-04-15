# Perspective Taking in Deep Reinforcement Learning Agents
## Introduction
In this repository, you can find the code used for [Perspective taking in deep reinforcement learning agents (coming soon)](https://arxiv.org/).

We used APES environment to build a task that need perspective taking to be solved. The task consist of two agents *Subordinate* and *Dominant*. The task also include a food that the *Subordinate* should eat if it is not observed by the *Dominant*. The *Subordinate* was driven by a neural network trained using reinforcement learning.
## Installation
The instruction for installation will be using a conda.
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
To train the models you need to take a look at File ```bash.sh```
### Load pre-trained models:
You can download the pre-trained models from [here](https://drive.google.com/file/d/1yQqOxCPAhApKMj0nZT1BT03hluNy-zDz/view?usp=sharing) 

## Simulations
### Re-generate the simulations
To re-do the simulations, you need to have the training models (downloaded or re-trained). 
### Download simulations
All simulation to execute "Stage 2" jupyter notebook can be downloaded from [here](https://drive.google.com/file/d/1c_UZAYQdGAnRIkH21zfsMktGPtpzRCUj/view?usp=sharing). You need to extract the zipped file into "NPZ" folder. 
