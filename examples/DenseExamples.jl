using Pkg, CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;


using LinearAlgebra, NNlib, ComponentArrays, DifferentialEquations, CUDA


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
nObs = 4000

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 2, maxCoherence = .99) #sample from the basis
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
l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; tc = 1.0f0, α = 0.01f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2;  tc = 1.0f0, α = 0.01f0)

#initialize module and network
mo = CompositeModule(l0, (l1, l2))
pcn = PCNet(mo)

#optionally initialize first layer to known basis just to see how the forward pass works
#=
pcn.odemodule.layers[1].ps .= w
pcn.odemodule.layers[2].ps .*= 0
pcn.odemodule.layers[2].ps[diagind(pcn.odemodule.layers[2].ps)] .= 1.0f0
pcn.odemodule.layers[1].initializer!.ps .*= 0
pcn.odemodule.layers[2].initializer!.ps .*= 0
=#

########## Running the network on the input x ##########
@time pcn(x,  (0.0f0, 55.0f0), abstol = 0.015f0, reltol = 0.1f0);


#view the results
obs = 1 #choose an observation to plot
yh = pcn.sol.u[1]
scatterlines(yh.L1[:, obs]) #plot the solution for layer L1

#plot the input compared to the predictions
f = scatterlines(x[:, obs])
scatterlines!(pcn.odemodule.predictions.L0[:, obs])
f



# Train the network in an unsupervised manner
reset!(pcn) #set all errors and predictions to 0
@time train!(pcn, x, (0.0f0, 50.0f0); iters = 1000, abstol = 0.01f0, reltol = 0.01f0, stops = 48.0f0:1.0f0:50.0f0)


########## Move to GPU and repeat ##########

to_gpu!(pcn)
xc = cu(x)

#running the network
@time pcn(xc, (0.0f0, 50.0f0), abstol = 0.01f0, reltol = 0.01f0);

#viewing the results
obs = 1
yh = pcn.sol.u[end]
scatterlines(Array(yh.L1)[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
f


########## Training with unsupervised learning ##########

reset!(pcn) #set all errors and predictions to 0
@time train!(pcn, xc, (0.0f0, 35.0f0); iters = 500, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.01f0, stops = 25.0f0:1.0f0:35.0f0)

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

to_cpu!(pcn)


size(aa, ndims(aa)) 