# Interaction-Network-Pytorch
Code adapted from [here](https://github.com/higgsfield/interaction_network_pytorch)

## Dependecies
- Python3
- Miniconda3/Anaconda3
- Pytorch

## Getting started
1. Clone this repo:
`git clone git@github.com:sanjass/IntNetworks6885.git`
2. Create Conda environment:
`conda env create -f environment.yml`
3. Make sure you are cd'ed into the repo's root and run:
 ` . ./start.sh` to activate the environment and set the PYTHONPATH.
4. Run model for training:
`python3 int_network.py`

## IntPhys data processing pipeline

Located in data_processing.py.
Saves a numpy matrix of shape (15000, 100, 5, 17) which is num_videos x num_frames x max_num_objects x max_num_features
--------------------------------------------------

Pytorch Implementraion of [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222).

#### Interaction Network is a powerful graph based framework for dynamic systems. It is able to simulate the physical trajectories of n-body, bouncing ball, and non-rigid string systems accurately over thousands of time steps, after training only on single step predictions.

*Our results provide surprisingly strong evidence of IN’s ability to learn accurate physical simulations and generalize their training to novel systems with different numbers and configurations of objects and relations… Our interaction network implementation is the first learnable physics engine than can scale up to real-world problems, and is a promising template for new AI approaches to reasoning about other physical and mechanical systems, scene understanding, social perception, hierarchical planning, and analogical reasoning.* 

Also see nice [blog post](https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/) and [keynote](https://www.slideshare.net/KenKuroki/interaction-networks-for-learning-about-objects-relations-and-physics).

Thanks to [jaesik817 tensorflow implementation](https://github.com/jaesik817/Interaction-networks_tensorflow) for Physics Engine. 
