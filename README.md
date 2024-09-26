# Toolbox of physics-inspired and physics-informed ML algorithms

Here, I include implementations of ML algorithms that I find useful for physics applications. 
This list is heavily biased by problems I've engaged with in the past, and so your favorites may be missing.
Rest assured I'll do my best to add to this as I encounter more new and exciting methods which are always proliferating.

## Directory Organization
```
├── env.yaml                          : Conda setup file with package requirements
├── README.md                         : Readme documentation
├── physics_ml_toolbox
    ├── fno                           : Fourier neural operators (and how to interpret them)
    ├── pinn                          : Physics-informed neural networks
    ├── pbnn                          : Physical bottleneck for PDE-constrained optimization
    ├── sindy                         : Sparse identification of nonlinear dynamical systems
```

This toolbox is not meant to be a "universal import" and I don't expect it to be a plug-and-play solution for general problems.
It should instead be viewed as a reference containing templates, sketches, and skeletons to be adapted to specific domains.
My hope is that by centralizing all of the various techniques I've implemented (and often discarded) across many, *many* git repositories,
I can minimize the amount of time I spend redoing solved problems or searching through old code.

Within each subfolder, I'll include an additional `README.md` with references to relevant papers. I'll also include any comments 
and, if possible, support those comments with suitable examples.
Each subfolder will also have its own python environment file. This is to minimize any headaches that may arise with resolving dependencies across multiple conflicting, old, or poorly maintained packages.

Please direct any questions, comments, or complaints to `jcolen@odu.edu`