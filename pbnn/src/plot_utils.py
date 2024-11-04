import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_mesh(data, mesh, ax,
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

def plot_grid(data, coords, ax,
              cmap=plt.cm.viridis,
              vmin=None, vmax=None,
              colorbar=True,
              colorbar_title=""):
    xmin, ymin = coords[0].min(), coords[1].min()
    xmax, ymax = coords[0].max(), coords[1].max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax.pcolormesh(coords[0], coords[1], data, norm=norm, cmap=cmap)
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