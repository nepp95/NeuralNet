# Neural Networks

Playground for learning about training networks. The first two examples can be viewed at tag `single-and-multi-neuron-network`, which isn't very efficient, but I had to start somewhere since I started with pretty much no knowledge on this subject. After that I used some abstraction and laid out data in a much more optimal way.

The old OpenGL implementation (network core, examples and visualisations) has been removed; the visualisation code was AI-written and intentionally kept separate from the network core. This repo is now being rebuilt on top of [EppoCore](https://github.com/nepp95/EppoCore), a Vulkan application core, so the next network implementation can use compute and graphics on a clean base. The old code remains available in the git history.

