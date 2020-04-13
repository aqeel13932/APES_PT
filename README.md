# Perspective Taking in Deep Reinforcement Learning Agents
## Introduction
The code in this repository was used for this [Perspective taking in deep reinforcement learning agents (coming soon)](https://arxiv.org/). We used APES environment to build a task that need perspective taking to be solved. The task consist of two agents *Subordinate* and *Dominant*. The task also include a food that the *Subordinate* should eat if it is not observed by the *Dominant*. The *Subordinate* was driven by a neural network trained using reinforcement learning.
## Installation
The instruction for installation will be using a conda.
```
# Create an environment and install the required packages.
conda create -n APES_PT python=3 keras=2.0.8 tensorflow=1.10.0 jupyter scikit-learn scikit-images
```
Some packages are not available in conda that we need to install them manually
```
# Activate the enviornment created
conda activate APES_PT
pip install scikit-video
```
## Training models or loading pretrained ones
### Training models: 
To train the models you need to take a look at File ```bash.sh```
### load pre-trained models
To load a pre-trained models you can use...
## Generating simulations
To Generate all required simulations, you need to 
