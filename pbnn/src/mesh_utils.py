import torch
import numpy as np

import dolfin as dlf
import dolfin_adjoint as d_ad

from scipy.interpolate import griddata

def scalar_img_to_mesh(img, x, y, function_space, vals_only=True, use_torch=True):
    ''' Nearest neighbor interpolation on a mesh '''
    dof_coords = function_space.tabulate_dof_coordinates().copy() # [N, 2]

    if use_torch:
        dof_coords = torch.tensor(dof_coords, dtype=img.dtype, device=img.device)
        xmin, ymin = x.min(), y.min()
        dx = x[1,1]-x[0,0]
        dy = y[1,1]-y[0,0]
        dof_x = (dof_coords[:, 0] - xmin) / dx
        dof_y = (dof_coords[:, 1] - ymin) / dy
        
        mesh_vals = img[dof_y.long(), dof_x.long()]
    else:
        mesh_vals = griddata((x.flatten(), y.flatten()), img.flatten(), dof_coords, method='nearest')
    
    if vals_only: 
        return mesh_vals

    mesh_func = d_ad.Function(function_space)
    if use_torch:
        mesh_func.vector()[:] = mesh_vals.detach().cpu().numpy()
    else:
        mesh_func.vector()[:] = mesh_vals
    
    return mesh_func