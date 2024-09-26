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
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.01f0);
pcmoduleE = PCModule(l0, (l1,))

#assign the true basis to layer L1's model parameters to analyze the convergence of the forward pass with a fully-trained network
pcmoduleE.ps.params.L1 .= w

# create a copy of the module for the Adam solver
pcmoduleA = deepcopy(pcmoduleE)



# Let's create solvers for Euler's method and Adam's method for the forward pass, and Euler's method for the backward pass (which we won't do here)
eulerf = ForwardEulerSolver(pcmoduleE, dt = 0.15f0)
adamf = ForwardAdamSolver(pcmoduleA; α = 0.01235f0, β1 = 0.25f0, β2 = 0.35f0) #note that Adam needs a completely different tuning on the forward pass than on the backward pass. It can be tricky to tune!
eulerb = BackwardEulerSolver(pcmoduleE, dt = 0.01f0)
adamb = BackwardAdamSolver(pcmoduleA; α = 0.01f0, β1 = 0.8f0, β2 = 0.99f0)


pcnEuler = PCNetwork(pcmoduleE, eulerf, eulerb)
pcnAdam = PCNetwork(pcmoduleA, adamf, adamb)

# Put the input data into a named tuple where the keys are the names of the layers to be clamped and the values are the values the layers are clamped to 
input = NamedTuple((L0 = x,))

@time pcnEuler(input; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)
@time pcnAdam(input; maxIters = 100, stoppingCondition = 0.01f0, reinit = true) # Adam is much faster with these hyperparameters!


obs = 2 # which observation to look at
f = scatterlines(y[:, obs]) #plot the true encodings
scatterlines!(get_states(pcnEuler).L1[:, obs]) # plot Euler's encodings
f
scatterlines!(get_states(pcnAdam).L1[:, obs]) # plot Adam's encodings
f # Not only is Adam faster, but it also converges to a better solution!




# Now let's look at the predicted inputs vs. the true inputs. Again, we want these to be similar
f = scatterlines(x[:, obs])
scatterlines!(get_predictions(pcnEuler).L0[:, obs])
f
scatterlines!(get_predictions(pcnAdam).L0[:, obs])
f

# Now let's look at the norms of the error vectors and update vectors for each iteration
# When choosing a solver and hyperparameters, we want to see these steadily decrease over time and
# flatten out towards the end in as few steps as possible. We may have to play around with solver 
# parameters, maxIters, stoppingCondition, and the solver type to achieve this.

# If these are rough and zig-zaggy, the step size is too large. If it isn't steep early on and never
# flattens out, then either step size is too small, the stopping condition is too large, or maxIters is 
# too small. Note that with a properly tuned step size, convergence should happen in 30 iterations or less
# with Euler's method.

scatterlines(get_du_logs(pcnEuler)) 
scatterlines(get_du_logs(pcnAdam)) 
scatterlines(get_error_logs(pcnEuler))
scatterlines(get_error_logs(pcnAdam)) 



pcnEuler.pcmodule.ps.initps .= 0
trajectories = log_trajectories(pcnAdam, NamedTuple((L0 = x,)); maxIters = 500, stoppingCondition = 0.001f0, reinit = true)
trajectories[1]

function plot_trajectories(trajectories)
    fig = Figure()
    ax = Makie.Axis(fig, xlabel = "Time", ylabel = "State")
    for i in eachrow(trajectories)
        lines!(i)
    end
    fig[1, 1] = ax
    fig
end

plot_trajectories(trajectories[1])