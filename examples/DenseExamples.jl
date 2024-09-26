#using Pkg# , CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
#Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;



using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, CairoMakie
using Flux: Flux, DataLoader

# Generate synthetic data from a given basis. The bases are discretely sampled Gaussians,
# which is common in image and audio processing. These are 1D but could easily be generalized
# to higher dimensions. 

# The goal is to sparsely encode this data. Each input will contain a small number of active bases,
# and we want to determine the correct activities of each basis for a given input. This inverse 
# problem is the fundamental multivariate problem that nearly all neural networks aim to solve, 
# regardless of the type of input signal or network architecture. 

# The chosen basis is extremely coherent, meaning that they are highly correlated. Sparsely encoding 
# such signals remains an open problem, so hopefully these networks can accomplish it. We will first assume
# the bases are known, i.e., the network is already trained. Later we'll see how well these networks can 
# learn the bases



n = 64; #number of bases
m = 64; 
nObs = 1024 * 4

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 6, maxCoherence = .25) #sample from the basis
y


heatmap(w)

f = scatterlines(y[:, 1]);
scatterlines!(x[:, 1]);

f


########## Initialize the network ##########

#layer dimensions, must be a tuple
n0 = (n, 1)

n1 = (m, 1)
n2 = (m, 1)

#initialize layers


l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.1f0);
mo = PCModule(l0, (l1,))

# Assign the true basis to layer L1's model parameters to analyze the convergence of the forward pass with a fully-trained network
mo.ps.params.L1 .= w

# Create the solvers
fSolver = ForwardEulerSolver(mo, dt = 0.15f0)
bSolver = BackwardEulerSolver(mo, dt = 0.01f0)

pcn = PCNetwork(mo, fSolver, bSolver)



########## Running the network with the known optimal weights, unsupervised ##########


# play around with maxIters, stoppingCondition, and dt of the forward solver to see how the network converges
change_step_size_forward!(pcn, (dt = 0.2f0,))
input = NamedTuple((L0 = x,))

@time pcn(input; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)
get_states(pcn).L1

obs = 2 # which observation to look at
f = scatterlines(get_states(pcn).L1[:, obs])
scatterlines!(y[:, obs])
f

scatterlines!(x[:, obs])
f

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))


########## Running the network with the known optimal weights, supervised ##########
input = NamedTuple((L0 = x, L1 = y))
@time pcn(input; maxIters = 100, stoppingCondition = 0.000000001f0, reinit = true)
get_states(pcn)

f = scatterlines(get_states(pcn).L1[:, 1])

scatterlines!(y[:, 1])
f

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))


########## Running the network with the known optimal weights, unsupervised, on the GPU ##########

to_gpu!(pcn); #move to GPU 
input = NamedTuple((L0 = cu(x),)) #move data to GPU 


@time pcn(input; maxIters = 50, stoppingCondition = 0.01f0, reinit = true)

to_cpu!(pcn)
get_states(pcn)

obs = 2
f = scatterlines(get_states(pcn).L1[:, obs])
scatterlines!(y[:, obs])
f
scatterlines(get_u0(pcn).L1[:, obs])

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))
heatmap(get_model_parameters(pcn).L1)



########## Training with unsupervised learning ##########

# make a much larger data set
n = 64; #number of bases
m = 64; 
nObs = 1024 * 48

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 3, maxCoherence = .7) #sample from the basis
y



#layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)
#n2 = (m, 1)

#initialize layers


l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.0f0);

mo = PCModule(l0, (l1,))


fSolver = ForwardEulerSolver(mo, dt = 0.1f0)
bSolver = BackwardEulerSolver(mo, dt = 0.1f0)

pcn = PCNetwork(mo, fSolver, bSolver)
to_gpu!(pcn)


batchSize = 1024 * 8
trainingData = Flux.DataLoader((L0 = x,), batchsize = batchSize, partial = false, shuffle = true)

@time train_unsupervised!(pcn, trainingData; maxIters = 75, stoppingCondition = 0.0025f0, epochs = 20, followUpRuns = 2000, maxFollowUpIters = 5, print_batch_error = false)
change_step_size_backward!(pcn, (dt = 0.05f0,))
# look at the convergence of the training algorithm
scatterlines(get_training_du_logs(pcn))
scatterlines(get_training_error_logs(pcn))


#back to CPU for analysis
to_cpu!(pcn)

# layer 1 parameters. If this looks like a Gaussian basis, then the network has learned the true basis (up to a permutation)!
heatmap(get_model_parameters(pcn).L1)

# layer 1 initializer parameters. This should be highly sparse, incoherent, and correlated with the layer 1 parameters.
heatmap(get_initializer_parameters(pcn).L1')


# run the newly trained network
@time pcn(x; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)




# compare u0 to u. If these are similar, the initializer network is well trained and will converge quickly on the forward pass.
# note that u almost certainly won't look like the true activations y even when the network is well trained - this is unsupervised learning after all.

obs = 3
scatterlines(get_u0(pcn).L1[:, obs])
scatterlines(get_states(pcn).L1[:, obs])

# compare the predicted data to the true data. If these are similar, the network is well trained.
scatterlines(x[:, obs])
scatterlines(get_predictions(pcn).L0[:, obs])

#evaluate the convergence of the training algorithm
scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))

# run the newly trained network
@time pcn(x; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)





########## Training with supervised learning ##########

# make a much larger data set
n = 64; #number of bases
m = 64; 
nObs = 1024 * 24

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 3, maxCoherence = .7) #sample from the basis
y



# layer dimensions, must be a tuple
n0 = (n, 1)
n1 = (m, 1)
#n2 = (m, 1)

# create layers
l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.1f0);
mo = PCModule(l0, (l1,))


fSolver = ForwardEulerSolver(mo, dt = 0.1f0)
bSolver = BackwardEulerSolver(mo, dt = 0.5f0)
pcn = PCNetwork(mo, fSolver, bSolver)
to_gpu!(pcn)


batchSize = 1024 *8
trainingData = DataLoader((data = x, label = y), batchsize = batchSize, partial = false, shuffle = true)


@time train_supervised!(pcn, trainingData; maxIters = 50, stoppingCondition = 0.01f0, epochs = 100, followUpRuns = 1000, maxFollowUpIters = 5)
change_step_size_backward!(pcn, (dt = 0.9f0,))
# look at the convergence of the training algorithm
scatterlines(get_training_du_logs(pcn)[5:end])
scatterlines(get_training_error_logs(pcn))


#back to CPU for analysis
to_cpu!(pcn)

# layer 1 parameters. This should be almost identical to w, the true basis used to generate the data
heatmap(get_model_parameters(pcn).L1)

# layer 1 initializer parameters. What a beautiful matrix!
heatmap(get_initializer_parameters(pcn).L1')


# run the newly trained network
@time pcn(x; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)



# compare u0 to u. If these are similar, the initializer network is well trained and will converge quickly on the forward pass.
scatterlines(get_u0(pcn).L1[:, 1])
scatterlines(get_states(pcn).L1[:, 1])

scatterlines(y[:, 1]) # Wow! The network has learned the true activations y!

# compare the predicted data to the true data. If these are similar, the network is well trained.
scatterlines(x[:, 1])
scatterlines(get_predictions(pcn).L0[:, 1])


#evaluate the convergence of the training algorithm

duLogsWithoutInitializer = get_du_logs(pcn)
errorLogsWithoutInitializer = get_error_logs(pcn)
itersWithoutInitializer = get_iters(pcn)



# run the network again, this time with the initializer
@time pcn(x; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)

duLogsWithInitializer = get_du_logs(pcn)
errorLogsWithInitializer = get_error_logs(pcn)
itersWithInitializer = get_iters(pcn)

# the network converges much faster with the initializer!
scatterlines(duLogsWithoutInitializer)
scatterlines(duLogsWithInitializer)

scatterlines(get_states(pcn).L1[:, 1])
