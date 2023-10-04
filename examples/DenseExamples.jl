using Pkg, CairoMakie
cd("NeurodynamicalSystems"); #navigate to the package directory
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;


using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, DifferentialEquations, CUDA, MLDatasets


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
nObs = 1024

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 16, maxCoherence = .99) #sample from the basis
y


heatmap(w)

f = scatterlines(y[:, 1]);
scatterlines!(x[:, 1]);

f


########## Initialize the network ##########

#layer sizes
n0 = m
n1 = 128
n2 = 64

#initialize layers
l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)

#initialize module and network
mo = CompositeModule(l0, (l1, l2))
pcn = PCNet(mo)
to_gpu!(pcn)#move to GPU 
xc = cu(x)#move to GPU 
reset!(pcn)

########## Running the untrained network ##########
@time pcn(xc, (0.0f0, 50.0f0), abstol = 0.01f0, reltol = 0.01f0);

#viewing the results

obs = 10
yh = pcn.sol.u[end]
scatterlines(Array(yh.L1)[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
#scatterlines!(Array(pcn.odemodule.errors.L0)[:, obs])
f



f = scatterlines(Array(yh.L1)[:, obs])
scatterlines!(Array(pcn.odemodule.u0.L1)[:, obs])
#scatterlines!(Array(pcn.odemodule.errors.L0)[:, obs])
f

heatmap(Array(pcn.odemodule.layers[1].ps))
heatmap(Array(pcn.odemodule.layers[1].initializer!.ps))

########## Training with unsupervised learning ##########

reset!(pcn) #set all errors and predictions to 0
@time train!(pcn, xc, (0.0f0, 35.0f0); iters = 2500, decayLrEvery = 500, lrDecayRate = 0.9f0, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.01f0, stops = 25.0f0:1.0f0:35.0f0)

#viewing the results
obs = 1
yh = pcn.sol.u[end]
scatterlines(Array(yh.L1)[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
f


eachindex(pcn.odemodule.layers)

heatmap(Array(pcn.odemodule.layers[1].ps))
########## Training with supervised learning ##########

yc = cu(y)
@time pcn(xc, yc, (0.0f0, 50.0f0), abstol = 0.01f0, reltol = 0.01f0);

yc
pcn.sol.u[end].L2

obs = 1
yh = pcn.sol.u[end]
scatterlines(Array(yh.L2)[:, obs])
scatterlines(y[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
f



########## Testing on MNIST ##########
CUDA.memory_status()
x = yh = f = pcn = y = yc = xc = l0 = l1 = l2 = 0
GC.gc()

x = MNIST(Tx = Float32, split = :train) #load MNIST data
features = reshape(x.features, size(x.features, 1) * size(x.features, 2), size(x.features, 3)) #add the third (channel) dimension


#get the labels in a floating point one-hot matrix
labels = zeros(eltype(features), length(unique(x.targets)), length(x.targets))
for i in eachindex(x.targets)
    labels[x.targets[i] + 1, i] = 1.0f0
end

nObs = 512
ind = StatsBase.sample(1:size(features, 2), nObs, replace = false)


##### Initialize the network 

#layer sizes
n0 = size(features, 1)
n1 = 32
n2 = 10

#initialize layers
l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.01f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.01f0)

#initialize module and network
mo = CompositeModule(l0, (l1, l2))
pcn = PCNet(mo)
to_gpu!(pcn)#move to GPU 

x = cu(features[:, ind]) #move to GPU 
y = cu(labels[:, ind])

reset!(pcn)

########## Running the untrained network ##########
@time pcn(x, y, (0.0f0, 50.0f0), abstol = 0.01f0, reltol = 0.01f0);

obs = 1
yh = pcn.sol.u[end]
scatterlines(Array(yh.L2)[:, obs])

f = scatterlines(Array(x[:, obs]))
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
#scatterlines!(Array(pcn.odemodule.errors.L0)[:, obs])
f



########## Supervised Training ##########
reset!(pcn) #set all errors and predictions to 0
@time train!(pcn, x, y, (0.0f0, 35.0f0); iters = 1500, decayLrEvery = 100, lrDecayRate = 0.9f0, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.01f0, stops = 25.0f0:1.0f0:35.0f0)

mo.initerror.L1
mo.u0.L0 * mo.initerror.L1'
mo.layers[1].initializer!.ps

scatterlines(Array(mo.u0.L0)[:, 1])