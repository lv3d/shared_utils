import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch.nn.functional as F

voxformer_volume = np.load("kitt_3353_occupancy.npy")[...,-1].clip(0,1)
x_range = (0, 51.2)
y_range = (-25.6, 25.6)
z_range = (-2, 4.4)

# position = np.array([[2,4.2,4],[10,21,-0.8],[12,2.4,3.1]])
X = np.random.randint(low=x_range[0]+1,high=x_range[1],size=30)+0.1
Y = np.random.randint(low=y_range[0],high=y_range[1],size=30)+0.15
Z = np.random.randint(low=z_range[0]+0.2,high=z_range[1],size=30) +0.2
position = np.stack([X,Y,Z],axis=-1)

""scipy 实现的 Volume Trilinear interpolation"
x = np.linspace(x_range[0], x_range[1], 256 + 1)[:-1] + 0.1
y = np.linspace(y_range[0], y_range[1], 256 + 1)[:-1] + 0.1
z = np.linspace(z_range[0], z_range[1], 32 + 1)[:-1] + 0.1
fn = RegularGridInterpolator((x, y, z), voxformer_volume)
occupancy = fn(position).astype(np.float32)
print(f"GT occupancy:{occupancy}")

''' Torch 实现
Volume 的按照X,Y,Z的顺序 shape[256,256,32] 采样点的范围需要归一化 [-1,1]之间
按照 Torch 的官网介绍， Volume 需要转化成 （D,H,W）
'''
voxformer_volume = voxformer_volume.transpose(2,1,0)
voxformer_volume = torch.from_numpy(voxformer_volume)[None,None,...].double()


position = torch.from_numpy(position)
x_normalized = 2 * (position[..., 0] - x[0]) / (x[-1] - x[0]) - 1
y_normalized = position[..., 1] / y[-1]
z_normalized = 2 * (position[..., 2] - z[0]) / (z[-1] - z[0]) - 1
grid = torch.stack([x_normalized,y_normalized,z_normalized],dim=-1).double()
interpolated_values = F.grid_sample(voxformer_volume, grid[None,None,None,...], mode='bilinear', align_corners=True)
print(f"Torch 实现的: {interpolated_values}")


