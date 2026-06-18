# Computer-Vision-Assignment

All the information about the problem to solve is on this link: https://docs.google.com/document/d/1104N-4ZyQn7HBv8AHL63iC8jejaEInjRqH4Ijs7F5sY/edit?pli=1&tab=t.0#heading=h.b54a7hm8jvif

The whole idea to solve the problem is based on use Open3D as the main resource. First problem is to align the images and the trajectory in the same coordinate system expected by the viewer, using Iterative Closest Point(ICP) algorithm and rigidly transforms all the PLY and TXT file to infer the unknown difference between point clouds and camera poses, estimating the geometric transformation matrix. Then, return the final result as desired.
I used, as an example, the case for 2 images and extrapolate for 3.
I tried to implement Point-to-Plane ICP Registration algorithm to a faster convergence speed.

