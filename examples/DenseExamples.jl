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
nObs = 1024 * 8

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 3, maxCoherence = .99) #sample from the basis
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
l1 = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
#l2 = PCDense(n2, n1, relu, 0.1f0, :L2, Float32)
mo = PCModule(l0, (l1,))

pcn.mo.ps.params.L1 
mo.receptiveFieldNorms.L1


fSolver = forwardES1(mo, dt = 0.15f0)
bSolver = backwardES1(mo, dt = 0.01f0)

pcn = Pnet(mo, fSolver, bSolver)

#assign the true basis to layer L1's model parameters to analyze the convergence of the forward pass with a fully-trained network
pcn.mo.ps.params.L1 .= w
#pcn.mo.ps.params.L2 .= 0
#pcn.mo.ps.params.L2[diagind(pcn.mo.ps.params.L2)] .= 1


########## Running the network with the known optimal weights, unsupervised ##########


# play around with maxIters, stoppingCondition, and dt of the forward solver to see how the network converges
@time pcn(x; maxIters = 100, stoppingCondition = 0.001f0, use_neural_initializer = false, reset_states = true)
get_states(pcn)

obs = 2
f = scatterlines(get_states(pcn).L1[:, obs])
scatterlines!(y[:, obs])
f

scatterlines!(x[:, obs])
f

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))



########## Running the network with the known optimal weights, supervised ##########

@time pcn(x, y; maxIters = 100, stoppingCondition = 0.000000001f0, use_neural_initializer = true, reset_states = true)
get_states(pcn)

f = scatterlines(get_states(pcn).L1[:, 1])

scatterlines!(y[:, 1])
f

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))


########## Running the network with the known optimal weights, unsupervised, on the GPU ##########

to_gpu!(pcn); #move to GPU 
xc = cu(x) #move data to GPU 


@time pcn(xc; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)

to_cpu!(pcn)
get_states(pcn)

f = scatterlines(get_states(pcn).L1[:, 1])
scatterlines!(y[:, 1])
f
scatterlines(get_u0(pcn).L1[:, 1])

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))
heatmap(get_model_parameters(pcn).L1)



########## Training with unsupervised learning ##########

#layer dimensions, must be a tuple
n0 = (n, 1)

n1 = (m, 1)
n2 = (m, 1)

#initialize layers


l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
#l2 = PCDense(n2, n1, relu, 0.1f0, :L2, Float32)
mo = PCModule(l0, (l1,))


FfSolver = forwardES1(mo, dt = 0.1f0)
bSolver = backwardES1(mo, dt = 0.01f0)

pcn = Pnet(mo, fSolver, bSolver)


to_gpu!(pcn)
xc = cu(x)


batchSize = 1024 * 1
trainingData = DataLoader(xc, batchsize = batchSize, partial = false, shuffle = true)



@time trainSteps!(pcn, trainingData; maxIters = 150, stoppingCondition = 0.01f0, trainingSteps = 100, followUpRuns = 10, maxFollowUpIters = 10)


for xd in trainingData
    println(xd isa CuArray) 

end

pcn.mo.ps.params.L1 
pcn.mo.inputlayer.data
pcn.mo.du.L0


#back to CPU for analysis
to_cpu!(pcn)
scatterlines(get_error_logs(pcn))
scatterlines(get_du_logs(pcn))


# layer 1 parameters. If this looks like a Gaussian basis, then the network has learned the true basis (up to a permutation)!
heatmap(get_model_parameters(pcn).L1)

# layer 1 initializer parameters. This should be highly sparse, incoherent, and correlated with the layer 1 parameters.
heatmap(get_initializer_parameters(pcn).L1')

scatterlines(get_u0(pcn).L1[:, 1])
scatterlines(get_states(pcn).L1[:, 1])

scatterlines(y[:, 1])
scatterlines(x[:, 1])

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))
#evaluate the convergence of the training algorithm

scatterlines(get_training_du_logs(pcn))
scatterlines(get_training_error_logs(pcn))


@time pcn(x; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)





bs = 1024 
data = DataLoader(xc, batchsize = bs, partial = false, shuffle = true)
data.batchsize
GC.gc(true)


@time train!(pcn, data; maxSteps = 50, stoppingCondition = 0.01, maxFollowupSteps = 10, epochs = 500, trainstepsPerBatch = 20, decayLrEvery = 20, lrDecayRate = 0.85f0, show_every = 1, normalize_every = 1000)

to_cpu!(pcn)
heatmap(pcn.pcmodule.layers[1].ps)
heatmap(pcn.pcmodule.layers[2].ps)
heatmap(pcn.initializer!.initializers[1].ps)
to_gpu!(pcn)

@time sol = pcn(first(data), maxSteps = 150, stoppingCondition = 0.045f0, reset_module = true);
pcn.initializer!.isActive = true
pcn.pcmodule.u0.L0
pcn.initializer!(pcn.pcmodule.u0, first(data))
pcn.initializer!.isActive = false
to_cpu!(pcn)
sol = to_cpu!(sol)

obs = 888
f = scatterlines(sol.L2[:, obs]) #plot the output of the first layer
scatterlines!(pcn.pcmodule.u0.L2[:, obs])
f

sum(abs.(sol.L1))
sum(abs.(pcn.pcmodule.u0.L1 .- sol.L1))

scatterlines(pcn.fixedPointSolver.errorLogs)
scatterlines(pcn.fixedPointSolver.duLogs)
length(fps.duLogs)
minimum(fps.duLogs)




