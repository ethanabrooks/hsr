#! /usr/bin/env python
import os
import sys

from mujoco_py import load_model_from_path, MjSim, MjViewer

print(os.getcwd())

if __name__ == '__main__':
    model = load_model_from_path(sys.argv[1])
    sim = MjSim(model)
    viewer = MjViewer(sim)
    input()
    while True:
        viewer.render()
        sim.step()
