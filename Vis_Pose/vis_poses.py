import numpy as np
import os
from scipy.io import loadmat
from matplotlib import cm
import matplotlib.pyplot as plt

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(camera_matrix, width=0.64/2, height=0.48/2, scale_focal=40, draw_frame_axis=False):
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [-width, -height, f_scale]
    X_triangle[0:3,1] = [0, -2*height, f_scale]
    X_triangle[0:3,2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def visualize_axis(RTs, radius=10.0, vertices=None):
   """
   params:
   RTs: camera poses, Nx4x4
   radius: control the side lens of the axes for each camera
   vertices: optional, for visualization of the points
   """
   fig = plt.figure()
   ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
   ax_3d.grid(False)
   ax_3d.set_xlabel('X')
   ax_3d.set_ylabel('Y')
   ax_3d.set_zlabel('Z')

   radius = radius
   sphere = np.random.randn(3, 100)
   sphere = radius * sphere / np.linalg.norm(sphere, axis=0, keepdims=True)
   if vertices is None:
       ax_3d.scatter(*sphere, c='k', alpha=0.1)  # random points
   else:
       for vertex in vertices:
           ax_3d.scatter(*vertex.T, alpha=0.1)

   s = 0.1 * radius
   for RT in RTs:
       R = RT[:3, :3]
       T = RT[:3, 3]
       e1, e2, e3 = s * R.transpose(1, 0) + T.reshape(1, 3)
       ax_3d.plot(*np.stack([e1, T], axis=1), c='r')  # a line connecting point e1 and T, red
       ax_3d.plot(*np.stack([e2, T], axis=1), c='g')
       ax_3d.plot(*np.stack([e3, T], axis=1), c='b')
   ax_3d.set_xlim(-5, 5)  # set the axis limits
   ax_3d.set_ylim(-5, 5)
   # plt.show()
   plt.savefig('poses.png')


def visualize_cams(RTs, Ks, colors=None, pts=None, train_ratio=1.0, ax=None):
    """
    params:
    RTs: camera extrinsics, Nx4x4
    Ks: camera intrinsics, Nx3x3
    colors & train_ratio: different colors for cameras used for training and other purposes
    """
    R_world2cam = RTs[:, :3, :3]
    T_world2cam = RTs[:, :3, 3]

    R_cam = np.transpose(R_world2cam, (0, 2, 1))
    T_cam = []
    T_cam2 = []
    for idx in range(R_cam.shape[0]):
        T_cam_i = -np.matmul(R_cam[idx], T_world2cam[idx])
        T_cam.append(T_cam_i)
        T_cam2.append(T_cam_i + np.matmul(R_cam[idx], [1,0,0]))
    T_cam = np.asarray(T_cam)
    T_cam2 = np.asarray(T_cam2)

    if colors is None:
        cm_subsection = np.ones((RTs.shape[0],)) * 0.9
        cm_subsection[0:int(train_ratio*RTs.shape[0])] = 0.1
        colors = [ cm.magma(x) for x in cm_subsection ]
    else:
        assert colors.shape[0]==RTs.shape[0]

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    X_moving = create_camera_model(Ks[0])
    patternCentric = True
    for idx in range(RTs.shape[0]):
        cMo = RTs[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], patternCentric)
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
    max_range = np.array([T_cam[:,0].max()-T_cam[:,0].min(), T_cam[:,1].max()-T_cam[:,1].min(), T_cam[:,2].max()-T_cam[:,2].min()]).max() / 2.0
    max_range = max_range*1.2

    mid_x = (T_cam[:,0].max()+T_cam[:,0].min()) * 0.5
    mid_y = (T_cam[:,1].max()+T_cam[:,1].min()) * 0.5
    mid_z = (T_cam[:,2].max()+T_cam[:,2].min()) * 0.5
    #ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #ax.set_ylim(mid_z - max_range, mid_z + max_range)
    #ax.set_zlim(mid_y - max_range, mid_y + max_range)
    #ax.scatter(T_cam[:,0], T_cam[:,1], T_cam[:,2]);
    #ax.quiver(T_cam[:,0], T_cam[:,1], T_cam[:,2], T_cam2[:,0], T_cam2[:,1], T_cam2[:,2]);
    #ax.scatter(pts[::5,0], pts[::5,1], pts[::5,2], s=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # plt.show()
    plt.savefig('cams.png')


if __name__=='__main__':
    K_ = np.array([[64,      0,       32.0,],
                   [0,       64,      32.0,],
                   [0,       0,       1]])

    # load camera poses, change here to use your own data
    import json
    # json_path = '/data/jxhuang/datasets/rabitstamp/multiview_7x3_5x5_images_rgb/transforms.json'
    json_path = 'dataset.json'
    with open(json_path) as f:
        data = json.load(f)
    # poses = np.array([item['transform_matrix'] for item in data['frames']])  # (N, 4, 4)
    poses = np.array([v for k, v in data['poses'].items()])  # (N, 3, 4)
    poses = np.concatenate((poses, np.tile(np.array([0., 0, 0, 1]).reshape(1, 1, 4), (poses.shape[0], 1, 1))), axis=1)  # (N, 4, 4)
    
    world2cams = np.linalg.inv(poses)
    K = np.tile(K_.reshape(-1, 3, 3), (poses.shape[0], 1, 1))
    visualize_axis(poses, radius=2.0)  # visualize camera poses with rgb axes 
    visualize_cams(world2cams, K)  # visualize cameras with plane models
