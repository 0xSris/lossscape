import numpy as np
import matplotlib.pyplot as plt

from lossscape.surfaces import SURFACES
from lossscape.optimizers import SGD, Momentum, AdamLite
from lossscape.visualizers import Visualizer


from lossscape.surfaces import SURFACES
from lossscape.optimizers import SGD
from lossscape.visualizers import Visualizer

def main():
    surface = SURFACES[0]  # start with Quadratic
    optimizer = SGD(lr=0.1)
    viz = Visualizer(surface, optimizer, surfaces=SURFACES)
    viz.run()

if __name__ == "__main__":
    main()
