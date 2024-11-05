# Fourier neural operators for parametric partial differential equations

Fourier neural operators are a deep learning technique that aims to learn solution operators for parametric partial differential equations. That is, they seek to learn a mapping from an equations parameters (i.e. coefficients or initial conditions) to its solution. To accomplish this, they learn an integral kernel $k(\mathbf{x}, \mathbf{y})$ that is applied to feature fields $\phi(\mathbf{x})$ as

$$ \phi (\mathbf{x}) \rightarrow \int k(\mathbf{x} - \mathbf{x'}) \phi(\mathbf{x'})\, d\mathbf{x'} $$

The general neural operator aims to address a more broad series of kernel structuresa, but FNO assumes translational invariance which means the kernel $k(\mathbf{x}, \mathbf{y})$ depends only on the two-point separation $\mathbf{x} - \mathbf{y}$. This allows one to exploit the efficiency of the FFT algorithm via a spectral convolution

$$ \int k(\mathbf{x} - \mathbf{x'}) \phi(\mathbf{x'})\, d\mathbf{x'} = \mathcal{F}^{-1} \bigg( \tilde{k} \cdot \mathcal{F} \big( \phi(\mathbf{x} \big) \bigg) $$

FNO networks learn a series of integral kernels which generically approximate nonlinear solutions to partial differential equations. For more detail on this procedure and a derivation of this result, see the [FNO paper](https://arxiv.org/abs/2010.08895).

Practically, FNO enables long-range feature interaction. When using a CNN, features accumulate long-range information only if the receptive field is sufficiently large. This often requires a deep network with many convolutional layers. FNO can accumulate features over the entire input in a single layer. The tradeoff is that the spectral convolution can have trouble resolving short-wavelength features unless it keeps track of a large number of Fourier modes.


## Implementations

In this directory, we apply the Fourier neural operator method to solve:

- The 1-dimensional viscous Burgers' equation as both a 1D spatial and 1+1D spatiotemporal mapping problem
- The 2-dimensional Darcy flow equation (or steady-state diffusion equation) as a 2D spatial mapping problem

Here, the Burgers' equation and Darcy flow datasets from their [Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-). This data is too big to git-track but you can download it, extract it, and point the data_processing scripts to the correct locations by modifying the configuration files.

In my own work, I have found this technique and variations useful for learning Green's functions from [microfluidic particle streams](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.107301) and [cell contractility experiments](https://www.cell.com/cell/abstract/S0092-8674(23)01331-4)