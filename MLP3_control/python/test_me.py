import pydart2 as pydart
from pydart2.world import World
from pydart2.skeleton import Skeleton

# Initialize DART
pydart.init()

# Load a world and a skeleton
world = World(0.001, 'path/to/your/world/file.skel')
skeleton = Skeleton('path/to/your/skeleton/file.skel')

# Add the skeleton to the world
world.add_skeleton(skeleton)

# Simulate the world for a certain number of steps
for i in range(1000):
    world.step()
    print(world.time(), skeleton.q)
