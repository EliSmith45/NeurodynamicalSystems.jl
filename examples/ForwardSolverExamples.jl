# comparison of the forward pass convergence behavior of different solvers

using Revise;
using NeurodynamicalSystems;



using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, CairoMakie
#using Flux: Flux, DataLoader


# Start with the same Gaussian basis used in DenseExamples.jl

n = 64; #number of bases
m = 64; 
nObs = 1024

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 10, maxCoherence = .25) #sample from the basis


# Create the network

# Layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)

# Create the layers
l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
mo = PCModule(l0, (l1,))

#assign the true basis to layer L1's model parameters to analyze the convergence of the forward pass with a fully-trained network
mo.ps.params.L1 .= w

# Let's start with Euler's method for inference (forward pass) and learning (backward pass)
fSolver = ForwardEulerSolver(mo, dt = 0.5f0)
bSolver = BackwardEulerSolver(mo, dt = 0.01f0)

pcnEuler = Pnet(mo, fSolver, bSolver)

# Run the network
@time pcnEuler(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)


# Plot the resulting encodings. These should be similar.
obs = 2 # which observation to look at
f = scatterlines(get_states(pcnEuler).L1[:, obs])
scatterlines!(y[:, obs])
f

# Now let's look at the predicted inputs vs. the true inputs. Again, we want these to be similar
f = scatterlines(pcnEuler.mo.predictions.L0[:, obs])
scatterlines!(x[:, obs])
f

# Now let's look at the norms of the error vectors and update vectors for each iteration
# When choosing a solver and hyperparameters, we want to see these steadily decrease over time and
# flatten out towards the end in as few steps as possible. We may have to play around with solver 
# parameters, maxIters, stoppingCondition, and the solver type to achieve this.

# If these are rough and zig-zaggy, the step size is too large. If it isn't steep early on and never
# flattens out, then either step size is too small, the stopping condition is too large, or maxIters is 
# too small. Note that with a properly tuned step size, convergence should happen in 30 iterations or less
# with Euler's method.

scatterlines(get_du_logs(pcnEuler)) # we can see that the step size is too large!
scatterlines(get_error_logs(pcnEuler))




# Let's try changing the step size
change_step_size_forward!(pcnEuler, (dt = 0.01f0,))
@time pcnEuler(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)


# Plot the resulting encodings. These should be similar.
obs = 2 # which observation to look at
f = scatterlines(get_states(pcnEuler).L1[:, obs])
scatterlines!(y[:, obs])
f

# Now let's look at the predicted inputs vs. the true inputs. Again, we want these to be similar
f = scatterlines(pcnEuler.mo.predictions.L0[:, obs])
scatterlines!(x[:, obs])
f

scatterlines(get_du_logs(pcnEuler)) # now, the step size is too small. The norm of du should always decrease, and we should never need 100 iterations to converge
scatterlines(get_error_logs(pcnEuler)) 






# Let's try increasing the step size a little and making the stopping condition more strict
change_step_size_forward!(pcnEuler, (dt = 0.2f0,))
@time pcnEuler(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)


# Plot the resulting encodings. These should be similar.
obs = 2 # which observation to look at
f = scatterlines(get_states(pcnEuler).L1[:, obs])
scatterlines!(y[:, obs])
f

# Now let's look at the predicted inputs vs. the true inputs. Again, we want these to be similar
f = scatterlines(pcnEuler.mo.predictions.L0[:, obs])
scatterlines!(x[:, obs])
f


scatterlines(get_du_logs(pcnEuler)) # this looks much better!
scatterlines(get_error_logs(pcnEuler)) 



##### # Now let's try Heun's method instead of Euler's.



# Layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)

# Create the layers
l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
mo = PCModule(l0, (l1,))

#assign the true basis to layer L1's model parameters to analyze the convergence of the forward pass with a fully-trained network
mo.ps.params.L1 .= w

# Let's start with Euler's method for inference (forward pass) and learning (backward pass)
fSolver = ForwardHeunSolver(mo, dt = 0.45)
bSolver = BackwardEulerSolver(mo, dt = 0.01f0)
pcnHeun = Pnet(mo, fSolver, bSolver)
change_step_size_forward!(pcnEuler, (dt = 0.45,))

# Run the networks with identical stopping conditions
@time pcnHeun(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
@time pcnEuler(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)


# Plot the resulting encodings. These should be similar.
obs = 2 # which observation to look at
f = scatterlines(get_states(pcnHeun).L1[:, obs])
scatterlines!(get_states(pcnEuler).L1[:, obs])
f



scatterlines!(y[:, obs])
f

# Now let's look at the predicted inputs vs. the true inputs. Again, we want these to be similar
f = scatterlines(pcnHeun.mo.predictions.L0[:, obs])
scatterlines!(x[:, obs])
f


scatterlines(get_du_logs(pcnHeun)) 
scatterlines(get_du_logs(pcnEuler)) # Heun's method is more stable than Euler's

scatterlines(get_error_logs(pcnHeun))
scatterlines(get_error_logs(pcnEuler)) # Heun's method converges faster than Euler's

