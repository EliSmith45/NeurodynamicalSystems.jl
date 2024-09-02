# comparison of the training convergence behavior of different solvers

using Revise;
using NeurodynamicalSystems;



using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, CairoMakie
using Flux: Flux, DataLoader


# Start with the same Gaussian basis used in DenseExamples.jl

n = 64; #number of bases
m = 64; 
nObs = 1024 * 8
sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis
x, y = sample_basis(w; nObs = nObs, nActive = 10, maxCoherence = .25) #sample from the basis
xc = cu(x)
yc = cu(y)

batchSize = 1024 *4
trainingData = DataLoader(xc, batchsize = batchSize, partial = false, shuffle = true)


# Create the network

##### Using Euler's method for the backward pass (traditional gradient descent) #####
# Layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)

# Create the layers
l0Euler = PCStaticInput(n0, :L0)
l1Euler = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
moEuler = PCModule(l0Euler, (l1Euler,))

# Let's start with Euler's method for inference (forward pass) and learning (backward pass)
fSolverEuler = ForwardEulerSolver(moEuler, dt = 0.05f0)
bSolverEuler = BackwardEulerSolver(moEuler, dt = 0.02f0)
pcnEuler = Pnet(moEuler, fSolverEuler, bSolverEuler)
to_gpu!(pcnEuler)

# Run the network
@time pcnEuler(xc; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)

#back to CPU to quickly check that the step size is appropriate
to_cpu!(pcnEuler)
scatterlines(get_du_logs(pcnEuler))
scatterlines(get_error_logs(pcnEuler))
to_gpu!(pcnEuler)




@time trainSteps!(pcnEuler, trainingData; maxIters = 50, stoppingCondition = 0.01f0, trainingSteps = 500, followUpRuns = 50, maxFollowUpIters = 5)
# look at the convergence of the training algorithm
scatterlines(get_training_du_logs(pcnEuler))
scatterlines(get_training_error_logs(pcnEuler)) # Is error still decreasing at the end? If so, training isn't done.


##### Using Heun's for the backward pass #####

# Layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)

# Create the layers
l0Heun = PCStaticInput(n0, :L0)
l1Heun = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
moHeun  = PCModule(l0Heun, (l1Heun,))

# Let's start with Euler's method for inference (forward pass) and learning (backward pass)
fSolverHeun = ForwardEulerSolver(moHeun, dt = 0.05f0)
bSolverHeun = BackwardHeunSolver(moHeun, dt = 0.02f0)
pcnHeun = Pnet(moHeun, fSolverHeun, bSolverHeun)
to_gpu!(pcnHeun)

# Run the network
@time pcnHeun(xc; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)

#back to CPU to quickly check that the step size is appropriate
to_cpu!(pcnHeun)
scatterlines(get_du_logs(pcnHeun))
scatterlines(get_error_logs(pcnHeun))
to_gpu!(pcnHeun)




@time trainSteps!(pcnHeun, trainingData; maxIters = 50, stoppingCondition = 0.01f0, trainingSteps = 500, followUpRuns = 50, maxFollowUpIters = 5)
# look at the convergence of the training algorithm
scatterlines(get_training_du_logs(pcnHeun))
scatterlines(get_training_du_logs(pcnEuler))

scatterlines(get_training_error_logs(pcnHeun)) # Is error still decreasing at the end? If so, training isn't done.
scatterlines(get_training_error_logs(pcnEuler)) # Is error still decreasing at the end? If so, training isn't done.


