Rainbow is all you need
=========================

This repo is part of the Wintersemester course `Reinforcement Learning <https://uni-tuebingen.de/de/271554>` in the University of Tübingen.
It implents several Reinforcement Learning Agents that are trained to play on the Gym Environment  `HockeyEnv <https://github.com/martius-lab/hockey-env>`_
 

We were inspired by the Rainbow paper `Rainbow <https://arxiv.org/abs/1710.02298>`_ and the following  `repo <https://github.com/Curt-Park/rainbow-is-all-you-need>`_.
All agents beat the deterministic agents reliably.

The PER agent reached 39th place in the tournament and the DDQN agent reached 71th place (out of 117 participants). You can see the tournament results in the .ods file. Our agents are marked yellow.



Installation
------------
To install just install the requirements via pip. We advise to use a virtual environment.

Copy the weights outside the repo. Then you should be able to use our training to replicate our results.
We advise to use a GPU for training. 
Training can take about a day.
