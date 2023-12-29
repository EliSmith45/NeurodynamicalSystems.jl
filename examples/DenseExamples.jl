using Pkg, CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;



using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA
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
nObs = 1024 * 1

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 3, maxCoherence = .99) #sample from the basis
y


heatmap(w)

f = scatterlines(y[:, 1]);
scatterlines!(x[:, 1]);

f


########## Initialize the network ##########

#layer sizes
n0 = m
n1 = 64
n2 = 64

#initialize layers
l0 = PCStaticInput((n0, nObs), :L0)
l1, init1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)
l2, init2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)

#initialize module and network
mo, initializer = PCModule(l0, (l1, l2), (init1, init2))
fps = ODEIntegrator(mo; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false, dt = 0.05f0, adaptive = true, dtmax = 1.0f0, dtmin = 0.001f0)
pcn = PCNet(mo, initializer, fps)

pcn.pcmodule.layers[1].ps .= w
pcn.pcmodule.layers[2].ps .= zero(eltype(pcn.pcmodule.layers[2].ps))
pcn.pcmodule.layers[2].ps[diagind(pcn.pcmodule.layers[2].ps)] .= ones(eltype(pcn.pcmodule.layers[2].ps))



########## Running the untrained network ##########
to_cpu!(pcn)
@time sol = pcn(xx, maxSteps = 150, stoppingCondition = 0.05f0, reset_module = true);

scatterlines(pcn.fixedPointSolver.errorLogs)
scatterlines(pcn.fixedPointSolver.duLogs)
length(fps.duLogs)
minimum(fps.duLogs)


obs = 1
scatterlines(sol.L1[:, obs]) #plot the output of the first layer
scatterlines(sol.L2[:, obs]) #plot the output of the second layer
scatterlines(x[:, obs]) #plot the input data
scatterlines(y[:, obs]) #plot the targets
scatterlines(pcn.pcmodule.errors.L0[:, obs]) #plot the errors for the input layer

to_gpu!(pcn); #move to GPU 
xc = cu(x) #move data to GPU 
change_nObs!(pcn, 24)
xc = xc[:, 1:24]

pcn.fixedPointSolver.dt 
pcn.fixedPointSolver.c1
reset!(pcn)
@time solc = pcn(xc, maxSteps = 150, stoppingCondition = 0.05f0, reset_module = true); #run on GPU
solc = to_cpu!(solc) #move output to cpu for plotting




scatterlines(solc.L1[:, obs]) #plot the output of the first layer
scatterlines(solc.L2[:, obs]) #plot the output of the second layer

scatterlines(pcn.fixedPointSolver.errorLogs)
scatterlines(pcn.fixedPointSolver.duLogs)
length(fps.duLogs)
minimum(fps.duLogs)


########## Training with unsupervised learning ##########

n = 64; #number of bases
m = 64; 
nObs = 1024 * 4

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 5, maxCoherence = .99) #sample from the basis
y


#layer sizes
n0 = m
n1 = 64
n2 = 64

#initialize layers
l0 = PCStaticInput((n0, nObs), :L0)
l1, init1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.5f0)
l2, init2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.5f0)

#initialize module and network
mo, initializer = PCModule(l0, (l1, l2), (init1, init2))
fps = ODEIntegrator(mo; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false, dt = 0.05f0, adaptive = true, dtmax = 1.0f0, dtmin = 0.001f0)
pcn = PCNet(mo, initializer, fps)

to_gpu!(pcn)
xc = cu(x)


bs = 1024 
data = DataLoader(xc, batchsize = bs, partial = false, shuffle = true)
data.batchsize
GC.gc(true)


@time train!(pcn, data; maxSteps = 50, stoppingCondition = 0.05, maxFollowupSteps = 10, epochs = 100, trainstepsPerBatch = 20, decayLrEvery = 20, lrDecayRate = 0.85f0, show_every = 1, normalize_every = 1)

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
f = scatterlines(sol.L1[:, obs]) #plot the output of the first layer
scatterlines!(pcn.pcmodule.u0.L1[:, obs])
f

sum(abs.(sol))
sum(abs.(pcn.pcmodule.u0 .- sol))

scatterlines(pcn.fixedPointSolver.errorLogs)
scatterlines(pcn.fixedPointSolver.duLogs)
length(fps.duLogs)
minimum(fps.duLogs)




