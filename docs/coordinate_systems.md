# Coordinate system conventions

We frequently convert between different representations in this project, which can be challenging as each of them may use coordinates and memory layouts differently.
This document shall help by defining a minimal set of convention that all methods should follow.


## Memory layout and coordinates
The following applies to images and volumes.

- Neighboring voxels/pixels along the x-axis are neighbors in memory, i.e., the x-axis is the spatial axis along memory addresses change the slowest.
- Neighboring voxels/pixels along any other axis are not neighbors in memory.
- The distance in memory between voxels that are adjacent in y direction is smaller than for voxels that are adjacent in z direction.
- For all axes the memory address increases if we go into the positive direction of the respective axis

Following from this we associate the axes to array indices for C contiguous arrays like this `array[Z,Y,X]`. C contiguous is the default for numpy arrays.

![Memory layout and spatial axes](/media/memory_dims.png "Memory layout and spatial axes")


## Bounding boxes and voxel centers
We want to discretize a domain defined by an axis aligned box with a regular grid.
We agree on the following.

- Voxels are small cubes that have a well-defined extent and we store 1 value per voxel. 
- The centers of voxels do not lie on faces, edges, or corners of the bounding box. 
- All voxels are inside the bounding box and there is no empty space or overlap.

This leads to the following properties for our grids.
```
bbox_min = (x0,y0,z0)
bbox_max = (x1,y1,z1)
bbox_size = bbox_max - bbox_min

grid_size = resolution = (rx,ry,rz)
voxel_size = bbox_size/grid_size

The voxel center of the first voxel is
voxel_centers[0,0,0] = 0.5*voxel_size + bbox_min
```

![Bounding box with voxel centers example](/media/bounding_box_and_grid.png "Bounding box with voxel centers example")

