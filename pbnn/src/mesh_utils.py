import torch
import numpy as np

import dolfin as dlf
import dolfin_adjoint as d_ad

def convert_dof_array_to_function(dof_vals, function_space):
    if torch.is_tensor(dof_vals):
        dof_vals = dof_vals.detach().numpy()
    
    dof_func = d_ad.Function(function_space)
    dof_vals = dof_vals.T # Important to transpose because fenics does [N, C] but torch does [C, N]
    dof_func.vector().set_local(dof_vals.flatten())

    return dof_func

def function_space_dim(function_space):
    element_shape = function_space.ufl_element().value_shape()
    if not element_shape: #Empty
        return 1
    else:
        return np.prod(element_shape)

def multichannel_img_to_mesh(img, x, y, function_space, return_function=False):
    ''' Nearest neighbor interpolation on a mesh '''
    use_torch = torch.is_tensor(img)

    # For multi-channel i.e. higher-order inputs, dof coords groups vector elements together
    dof_coords = function_space.tabulate_dof_coordinates().copy() # [N, 2]

    if use_torch:
        dof_coords = torch.tensor(dof_coords, dtype=img.dtype, device=img.device)

    xmin, ymin = x.min(), y.min()
    dx = x[1,1]-x[0,0]
    dy = y[1,1]-y[0,0]
    dof_x = (dof_coords[:, 0] - xmin) / dx
    dof_y = (dof_coords[:, 1] - ymin) / dy

    if use_torch:
        dof_x = torch.round(dof_x).long()
        dof_y = torch.round(dof_y).long()
    else:
        dof_x = np.round(dof_x).astype(int)
        dof_y = np.round(dof_y).astype(int)
    
    skip = function_space_dim(function_space)
    dof_vals = [img[i, dof_y[i::skip], dof_x[i::skip]] for i in range(img.shape[0])]

    if use_torch:
        dof_vals = torch.stack(dof_vals)
    else:
        dof_vals = np.stack(dof_vals)
    
    if return_function:
        return convert_dof_array_to_function(dof_vals, function_space)
    else: 
        return dof_vals