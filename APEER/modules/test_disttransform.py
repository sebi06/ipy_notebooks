import numpy as np
import scipy.ndimage as nd


a = np.array(([0, 1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0, 0, 0]))

dt = nd.distance_transform_edt(a)
slices = nd.find_objects(input=a)
radii = [np.amax(dt[s]) for s in slices]

print(dt)
print(slices)
print(radii*2)
