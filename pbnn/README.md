# Physical bottleneck neural networks for PDE-constrained optimizaiton

The physical bottleneck approach was introduced in [this paper](https://www.cell.com/cell/abstract/S0092-8674(23)01331-4) to address the problem of estimating effective continuum parameters from measurement data.
The basic idea is to combine a deep neural network with a PDE solver, in this case the finite-element library [FEniCS](https://fenicsproject.org/). 
The network predicts parameters for a continuum mechanics problem, and then the PDE solver makes physically-constrained predictions using those parameters.
By backpropagating through the PDE solver to the neural network parameters, one can learn to estimate parameters that produce desired outputs, even when those parameters are not known *a priori*.

The directory is organized as follows
```
├── env.yaml                    : Conda setup file with package requirements
├── README.md                   : Readme documentation
├── configs                     : Training configuration files
├── models                      : Trained model weights for each dataset and model type
├── notebooks          
    ├── CreateDataset.ipynb     : Dataset creation for Poisson, Stokes, and Elastic Adhesion problems
    ├── DolfinAdjoint.ipynb     : Demo of forward/backward problems using `dolfin-adjoint` and `torch`
    ├── PBNNEvaluation.ipynb    : Evaluation of trained models for each dataset
    ├── TorchTesting.ipynb      : Comparing assembled `torch` problems to `dolfin` in each dataset     
├── src                         : Source code 
```

## The PBNN method

As a concrete example, consider the Poisson equation 

$$ \nabla^2 u = -f $$ 

The solution $u$ is determined by the forcing term $f$. Suppose we have an experiment where we can measure the outputs $u$ but cannot access $f$ - instead we have measurements of a related variable $\rho$. We don't know the relationship between $\rho$ and $u$, but we know (or at least strongly suspect) that $u$ obeys the Poisson equation. The physical bottleneck exploits this knowledge in order to find the mapping $\rho \rightarrow f$ that fits the data.

> [!NOTE]
>  ### How is this different than standard inverse problem solvers?
> Consider a typical forward problem containing measurements $u$, and a parametric model $F: f \rightarrow u$. Inverse problem solvers aim to find the optimal parameters $f$ such that $F(f) = u$. 
> 
> PBNN considers a slightly different setup, in which there are separate meaurements $\rho$ and $u$, a known mathematical model $F: f \rightarrow u$ and an unknown function $G: \rho \rightarrow f$. The objective is to learn the function $G$ such that $F[ G(\rho) ] = u$. 

A PBNN approach trains a neural network to learn this unknown mapping via the following steps

1. Collect a batch of inputs $\rho$ and targets $\hat{u}$
2. Predict $f = NN_{\theta} ( \rho  )$ using the neural network with weights $\theta$
3. Predict $u = F(f)$ using a PDE solver
4. Define an objective $\mathcal{L}(u, \hat{u})$ and compute parameter gradients $\frac{\partial \mathcal{L}}{\partial f}$ using the adjoint method (more below)
5. Compute neural network gradients using the chain rule/backpropagation $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial f} \frac{\partial f}{\partial \theta}$

After training completes, one can apply the trained network $NN_{\theta}$ to new measurements $\rho$ in order to predict the response $u$. 

In the original paper, the PBNN was applied to a biophysics problem. Here, one had mechanical measurements - a cellular traction force field $\mathbf{F}$ - and a proposed continuum model parameterized by a coefficient field $Y$. These coefficients were not known, nor could one access them in experiments. Instead, one could access relevant biochemical objects, such as the densities of focal adhesion proteins. The PBNN approach learned to map from these proteins to the model coefficients, and could generalize to predict mechanical behavior in previously unseen cells. 

> [!NOTE]
> ### How is this different from PINNs?
> Physics-informed neural networks optimize a joint objective that balances reconstruction and physics losses for a prediction $u$ and target $\hat{u}$ : 
> $$ \mathcal{L}_{PINN} = \underbrace{\langle  \left( u - \hat{u} \right)^2\rangle }_{\text{Reconstruction loss}} + \underbrace{ \langle \left( F(f) - u \right)^2 \rangle }_{\text{Physics loss}} $$
> PINNs may trade adherence to physics for predictive accuracy, which can pose problems when performing inverse parameter estimation. If the physics loss is large, then the parameters mean nothing as the model is incorrect. If the reconstruction loss is large, then the parameters mean nothing becaues they are not predictive. PINN-based parameter estimation requires finding the correct balance between these two objectives, which can be difficult for complex problems with noisy data. 
> 
> PBNN and PDE-constrained optimization in general circumvents this issue by **strictly enforcing** the physical model. In addition, PBNN can be applied for inference to new measurements, whereas PINNs must be retrained. 

## The adjoint method

The original PBNN implementation used the `pyadjoint` library to perform PDE-constrained optimization. This leveraged the adjoint method, which is explained briefly here following this [excellent tutorial](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf) by Andrew Bradley. 

Suppose we have a set of tunable parameters $f$ and values $u$, which are related by a known function $F(f, u) = 0$. The equation $F(f, u) = 0$ can be solved using a "forward solver" that determines the corresponding values of $u$ for a choice of parameters $f$. Our goal is to optimize an objective $G(u)$. To do this, we would like to differentiate our objective with respect to our tunable parameters, i.e. compute $\nabla_f G$. The **adjoint method** is a technique for doing this. 

First, we expand the gradient $\nabla_f G(u) = G_u \cdot  \nabla_f u$ using the chain rule. Here $G_u$ is the partial derivative of $G$ with respect to the field $u$. The second term $\nabla_f u$, or the derivative of the field $u$ w.r.t. the tunable parameters, comes from the solution of the forward problem $F$. Since $F(f, u) = 0$ everywhere, then $\nabla_f F(f, u) = 0$. Expanding this gives

$$ F_f + F_u \nabla_f u = 0 $$

We can rearrange and write $\nabla_f u = -F_u^{-1} \cdot F_f $ which can be substituted into our expression for $\nabla_f G$

$$ \nabla_f G(u) = -G_u \cdot F_u^{-1} \cdot F_f $$

The first two terms on the right hand side $ -G_u \cdot F_u^{-1}$ is the solution to the linear equation 

$$F_u^T \cdot \lambda + G_u = 0$$

Here the matrix transpose $F_u^T$ is called the adjoint (for real values) and the above equation is referred to as the **adjoint equation**. Solving this equation for the adjoint variables $\lambda = -G_u \cdot F_u^{-1}$ allows us to write

$$ \nabla_f G(u) = \lambda^T \cdot F_f $$

This turns out to be a computationally efficient technique for calculating gradients, because the Jacobian $ F_u$ is typically computed during the solution of the forward problem. 
For example, solving $F(f, u) = 0$ using Newton's method involves repeated update steps $\Delta u = - F / F_u $. 
Thus, using the transposed Jacobian to compute $ \nabla_f G(u)$ does not incur a significant additional cost.

An alternative technique for deriving the adjoint method uses Lagrange multipliers. Suppose we have the Lagrangian

$$ \mathcal{L} (f, u, \lambda) = G(u) + \lambda^T F(f, u) $$

Here $G(u)$ is the objective function and $\lambda$ is a vector of Lagrange multipliers enforcing the constraint $F(f, u) = 0$. One can see this by minimizing the Lagrangian by finding stationary points w.r.t. $\lambda$. 
$$ \nabla_{\lambda} \mathcal{L} = 0 = F(f, u) $$

Solving the first equation implies $\mathcal{L} (f, u, \lambda) = G(u)$, meaning that our target derivative is simply $\nabla_f \mathcal{L} $. We can compute this directly
$$ \nabla_f \mathcal{L} = G_u \nabla_f u + (\nabla_f \lambda^T) F(f, u) + \lambda^T ( F_u \nabla_f u + F_f ) $$
Since $F(f, u) = 0$, this simplifies to
$$ \nabla_f \mathcal{L} = G_u \nabla_f u + \lambda^T ( F_u \nabla_f u + F_f ) $$
$$ \nabla_f \mathcal{L} = (G_u + \lambda^T F_u) \nabla_f u + \lambda^T F_f $$

The expression within the parentheses is the linear equation for $\lambda$ defined above, i.e. the adjoint equation. When satisfied, the term drops out leaving $\nabla_f \mathcal{L} = \nabla_f G = \lambda^T F_f$ as before. 

## Converting the original pipeline to `torch`

A major drawback of the original PBNN method was its reliance on the `pyadjoint` and `dolfin-adjoint` libraries which lacked GPU support. As a result, the training loop had to continually move data to the GPU for neural network predictions, to the CPU for the PDE solver and adjoint-based gradient computation, and then back to the GPU to backpropagate gradients to the network weights. This was very slow. 

In this folder, I've made some progress towards streamlining the procedure and keeping the entire training loop on the GPU. On the toy datasets I consider, this has resulted in a modest ~4x speedup. I think there are more optimizations possible (particularly on the pre-computation and dataloading phase) but my goal here isn't to make a production-ready version. I just wanted to demonstrate how it could be done.

The basic idea is to convert the PDE forward problem to something that `pytorch` can handle and to let `autograd` handle the rest. Reverse-mode automatic differentiation computes the gradients w.r.t. parameters and these align with what `pyadjoint` gives (see `notebooks/DolfinAdjoint.ipynb`). 

To see how to do this in FEniCs, consider a typical problem written in the weak formulation

$$ \text{Find } u \in V \text{ s.t. } A(u, v) = L(v; f) \quad \forall\, v \in V$$

FEniCs assembles the bilinear form on the lhs into a matrix $\bar{A}$ and the linear form on the rhs into a vector $L$, and then solves the linear optimization problem $A u = L$. We can simply pre-compute these values, convert them to `torch` tensors, and solve the linear equation using `torch.linalg` functions. The linear form poses a slight wrinkle, as it typically contains information about the forcing term $f$ which is unknown *a priori*. To get around this, we can assemble a second linear problem $B (f, v) = L(v; f)$, extract the matrix $B$, and apply it to our discretized $f$. The overall procedure is roughly

1. Assemble the problems $A(u, v) = L(v; f_0)$, $B(f, v) = L(v; f_0)$ in FEniCs using a placeholder value $f_0$
2. Extract the matrices $A, B$ and convert them to `torch` tensors
3. Predict parameters using the neural networks $f = NN_{\theta}(\rho)$
4. Interpolate the predicted $f$ to the coordinates of the FEM degrees of freedom
5. Compute the linear form $L = B f$
6. Predict $u$ as the least squares solution to the linear equation $A u = L$ using `torch.linalg`
7. Compute the objective $\mathcal{L}(u, \hat{u})$ and backpropagate to compute network gradients

Training PBNNs through this procedure yields pretty much the same results (see the examples in this repo). This is not particularly groundbreaking. I'm just showing that we can take the "guts" out of the solver and get the same result without relying on external libraries like `pyadjoint` or `dolfin-adjoint`. In principle, this means we should be able to apply this approach generically to problems regardless of numerical implementation. Proving that is left as an exercise for another time.

Some additional comments:

- Computing $S = (A^T A)^{-1} A^T$ ahead of time speeds things up further. In this case Step 6 becomes $u = S L$
- The dof-interpolation in step 4 must be done differentiably. Right now, I only have nearest-neighbor interpolation
- Assembling each term in $A(u, v)$ separately (i.e. as $\{ A_1, A_2, A_3 \}$ allows you to handle learnable linear coefficients. In this case one has to predict the coefficients $\eta_i$ and then the full matrix $A = \sum \eta_i A_i$ 
- This technique as written is limited to linear problems. To handle nonlinear problems, one could 
    - Implement an alternative differentiable solution method for Step 6
    - Use a splitting scheme, see [here](https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/navier-stokes/python/documentation.html) for an example of how one might do this
- Dynamical problems are tricky as the gradients become very costly to track. A path forward could be integrating PBNN with [Neural ODEs](https://arxiv.org/abs/1806.07366)