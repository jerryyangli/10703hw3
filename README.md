# 10-703 Homework 3 Part 1 Problem 1 and Problem 2

Implementation of Reinforce, Actor-Critic

## Prerequisites

- Python 3.6
- Pytorch
- OpenAI Gym
- numpy
- scipy
- pybox2d
- gym[box2d]
- matplotlib
- pyglet
- h5py

## Installation

Please follow the instructions on the homework handout.

##  Testing

python reinforce.py --num-episodes <number of episodes> --lr <learning rate>


hw3_part1_plotter.py 


For example,

python reinforce.py --num-episodes 50000 --lr 5e-4

The above will run the REINFORCE algorithm for 50000 training episodes and for every 200 training episodes it will output the average test reward (over 100 episodes). The reward is outputted to console (and can be redirected to a file), and can be plotted with hw3_part1_plotter.py.



python a2c.py --num-episodes <number of episodes>  --lr <actor learning rate> --critic-lr <critic learning rate> --n <step N>


hw3_part2_plotter.py 


For example,

python a2c.py --num-episodes 50000 --lr 5e-4 --critic-lr 1e-4 --n 20

The above will run the advantage-actor critic algorithm for 50000 training episodes and for every 500 training episodes it will output the average test reward (over 100 episodes). The reward is outputted to console (and can be redirected to a file), and can be plotted with hw3_part2_plotter.py.  

