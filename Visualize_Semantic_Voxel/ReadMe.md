This Scripts can create 3D semantic voxel using Open3d, which contains the location and the color of the voxel grid.
1. Convert the Pointclouds to Voxelgrid.
2. The Voxel Grid initalize the pointcloud coordinate as the center of the Voxel Center.  
   **eg. if the voxel size = 1 and the point coordinate is [0,0,0], then the result voxel x range [-0.5,0.5], y and z is simliar with x coordinate.**
