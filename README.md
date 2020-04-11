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

Located in `data_processing.py`.

Saves a numpy matrix of shape `(15000, 100, 5, 17)` which is `num_videos x num_frames x max_num_objects x max_num_features`
The 17 features are:
```
# FORMAT for features of an object:
# [present_or_not (1 or 0),  - 1
# type of obj (occluder=1, non-occluder 0) -1
# location (x,y,z), - 3
# velocity (x,y,x), -3,
# scale (x,y,z), -3
# rotation (roll, pitch, yaw) -3,
# speed (only for occluders), -1
# friction, -1,
# mass (only for obj), -1] 
```
Index map:
- note that first two indices correspond to binary feature present-or-not and occluder-or-not
```
index_mapping = {"location_x":2, "location_y":3, "location_z":4, "velocity_x":5, "velocity_y":6, "velocity_z":7, "scale_x":8, "scale_y":9, "scale_z":10,
                 "rotation_roll":11, "rotation_pitch":12, "rotation_yaw":13, "speed":14, "friction":15, "mass":16}
```
## Example occluder and object from train data:
```
"occluder_2": {
                "location": {
                    "x": 572.5443115234375,
                    "y": 414.1805419921875,
                    "z": 0.0
                },
                "rotation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": -26.46286964416504
                },
                "scale": {
                    "x": 0.9712970852851868,
                    "y": 1.0,
                    "z": 2.3094539642333984
                },
                "friction": 0.5,
                "restitution": 0.5,
                "material": "M_Bricks_1",
                "speed": 3.1854804559914593,
                "moves": [
                    159,
                    192
                ]
            },
"object_1": {
                "location": {
                    "x": 453.3905029296875,
                    "y": -590.0372314453125,
                    "z": 54.33080291748047
                },
                "rotation": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": -46.122108459472656
                },
                "scale": {
                    "x": 1.086616039276123,
                    "y": 1.086616039276123,
                    "z": 1.086616039276123
                },
                "friction": 0.5,
                "restitution": 0.5,
                "material": "M_Tech_Hex_Tile",
                "mass": 131.9533233642578,
                "initial_force": {
                    "x": 21352.900390625,
                    "y": 38981.62109375,
                    "z": 24922.8046875
                },
                "velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "shape": "Sphere"
            }
```
--------------------------------------------------

Pytorch Implementraion of [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222).

#### Interaction Network is a powerful graph based framework for dynamic systems. It is able to simulate the physical trajectories of n-body, bouncing ball, and non-rigid string systems accurately over thousands of time steps, after training only on single step predictions.

*Our results provide surprisingly strong evidence of IN’s ability to learn accurate physical simulations and generalize their training to novel systems with different numbers and configurations of objects and relations… Our interaction network implementation is the first learnable physics engine than can scale up to real-world problems, and is a promising template for new AI approaches to reasoning about other physical and mechanical systems, scene understanding, social perception, hierarchical planning, and analogical reasoning.* 

Also see nice [blog post](https://blog.acolyer.org/2017/01/02/interaction-networks-for-learning-about-objects-relations-and-physics/) and [keynote](https://www.slideshare.net/KenKuroki/interaction-networks-for-learning-about-objects-relations-and-physics).

Thanks to [jaesik817 tensorflow implementation](https://github.com/jaesik817/Interaction-networks_tensorflow) for Physics Engine. 
