import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy.interpolate import griddata
from skimage.transform import downscale_local_mean

def plot_mesh(ax, data, mesh,
              cmap=plt.cm.viridis,
              vmin=None, vmax=None,
              colorbar=True,
              colorbar_title=""):

    # Dolfin mesh, stores values on vertices not faces
    coords = mesh.coordinates().T
    cells = mesh.cells().T
    values = data[cells].mean(axis=0)

    xmin, ymin = coords.min(axis=1)
    xmax, ymax = coords.max(axis=1)

    polygons = coords[:, cells]
    polygons = np.moveaxis(polygons, (0, 1, 2), (2, 1, 0))
    col = collections.PolyCollection(polygons)
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    col.set_color(cmap(norm(values)))
    ax.add_collection(col)
    ax.set(xlim=[xmin, xmax],
           ylim=[ymin, ymax])
    ax.set_aspect(1)

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        vmin, vmax = sm.get_clim()
        vmean = (vmin + vmax)/2
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        cbar = plt.colorbar(sm, cax=cax, ax=ax,
                            ticks=[vmin, vmean, vmax])
        cbar.ax.set_ylabel(colorbar_title, rotation=-90)

def plot_grid(ax, data, X, Y,
              cmap=plt.cm.viridis,
              vmin=None, vmax=None,
              colorbar=True,
              colorbar_title=""):
    xmin, ymin = X.min(), Y.min()
    xmax, ymax = X.max(), Y.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.pcolormesh(X, Y, data, norm=norm, cmap=cmap)
    ax.set(xlim=[xmin, xmax],
           ylim=[ymin, ymax])
    ax.set_aspect(1)

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        vmin, vmax = sm.get_clim()
        vmean = (vmin + vmax)/2
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        cbar = plt.colorbar(sm, cax=cax, ax=ax,
                            ticks=[vmin, vmean, vmax])
        cbar.ax.set_ylabel(colorbar_title, rotation=-90)

def plot_mesh_vector(ax, mesh_vals, mesh, N=100, **kwargs):
    X_mesh = mesh.coordinates()[:, 0]
    Y_mesh = mesh.coordinates()[:, 1]
    X_grid, Y_grid = np.meshgrid(
       np.linspace(X_mesh.min(), X_mesh.max(), N),
       np.linspace(Y_mesh.min(), Y_mesh.max(), N),
    )
    grid_vals = np.zeros([2, *X_grid.shape])

    grid_vals[0] = griddata((X_mesh, Y_mesh), mesh_vals[0], (X_grid, Y_grid))
    grid_vals[1] = griddata((X_mesh, Y_mesh), mesh_vals[1], (X_grid, Y_grid))

    plot_grid_vector(ax, 
                     grid_vals, 
                     X_grid, Y_grid, 
                     **kwargs)

def plot_grid_vector(ax, grid_vals, X_grid, Y_grid, 
                     downscale=4, threshold=0.1,             
                     cmap=plt.cm.viridis,
                     vmin=None, vmax=None,
                     colorbar=True,
                     colorbar_title=""):

    grid_norm = np.linalg.norm(grid_vals, axis=0)
    im = ax.pcolormesh(X_grid, Y_grid, grid_norm, vmin=vmin, vmax=vmax)

    grid_vals = np.stack([
        downscale_local_mean(grid_vals[0], (downscale, downscale)),
        downscale_local_mean(grid_vals[1], (downscale, downscale)),
    ])
    grid_norm = np.linalg.norm(grid_vals, axis=0)
    mask = grid_norm >= threshold

    X_grid = downscale_local_mean(X_grid, (downscale, downscale))
    Y_grid = downscale_local_mean(Y_grid, (downscale, downscale))

    ax.quiver(X_grid[mask], Y_grid[mask], grid_vals[0, mask], grid_vals[1, mask], color='white')

    if colorbar:
        vmin, vmax = im.get_clim()
        vmean = (vmin + vmax)/2
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        cbar = plt.colorbar(im, cax=cax, ax=ax,
                            ticks=[vmin, vmean, vmax])
        cbar.ax.set_ylabel(colorbar_title, rotation=-90)

def plot_with_regression(ax, x, y, color='black', label=''):
    score = r2_score(x, y)
    label = ' '.join([label, f'$R^2$ = {score:.2f}'])
    ax.scatter(x, y, color=color, label=label, s=5)

    model = LinearRegression().fit(x[:,None], y[:,None])
    score = model.score(x[:, None], y[:, None])
    eqn = f'y = {model.coef_[0,0]:.2f} x + {np.squeeze(model.intercept_):.2f}'

    xr = np.array([np.min(x), np.max(x)])[:, None]
    yr = model.predict(xr)
    ax.plot(xr, yr, color=color, label=eqn)