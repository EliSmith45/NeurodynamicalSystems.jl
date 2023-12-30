using Pkg, CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;


using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA, MLDatasets
using Flux: Flux, DataLoader


x = MNIST(Tx = Float32, split = :train) #load MNIST data
features = reshape(x.features, size(x.features, 1) * size(x.features, 2), size(x.features, 3)) #add the third (channel) dimension


#get the labels in a floating point one-hot matrix
labels = zeros(eltype(x.features), length(unique(x.targets)), length(x.targets))
for i in eachindex(x.targets)
    labels[x.targets[i] + 1, i] = 1.0f0
end

trainSize = 20000
trainInd = StatsBase.sample(1:size(features, 2), trainSize, replace = false)
testInd = setdiff(1:size(features, 2), trainInd)

trainFeatures = features[:, trainInd] #add the third (channel) dimension
trainLabels = labels[:, trainInd] #add the third (channel) dimension

##### Initialize the network 

#layer sizes
n0 = size(features, 1)
n1 = 32
n2 = 10


#initialize layers
l0 = PCStaticInput((n0, nObs), :L0)
l1, init1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.25, σ = relu, tc = 1.0f0, α = 0.5f0)
l2, init2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.0, σ = relu, tc = 1.0f0, α = 0.5f0)

#initialize module and network
mo, initializer = PCModule(l0, (l1, l2), (init1, init2))
fps = ODEIntegrator(mo; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false, dt = 0.05f0, adaptive = true, dtmax = 1.0f0, dtmin = 0.0001f0)
pcn = PCNet(mo, initializer, fps)

to_gpu!(pcn)
trainFeaturesc = cu(trainFeatures)
trainLabelsc = cu(trainLabels)

@time sol = pcn(trainFeaturesc[:, 1:128], trainLabelsc[:, 1:128], maxSteps = 10, stoppingCondition = 0.05f0, reset_module = true);



bs = 1024 
data = DataLoader((data = trainFeaturesc, label = trainLabelsc), batchsize = bs, partial = false, shuffle = true)
data.batchsize
GC.gc(true)


@time train!(pcn, data; maxSteps = 50, stoppingCondition = 0.05, maxFollowupSteps = 10, epochs = 500, trainstepsPerBatch = 15, decayLrEvery = 20, lrDecayRate = 0.85f0, show_every = 1, normalize_every = 1)

to_cpu!(pcn)
heatmap(pcn.pcmodule.layers[1].ps)
heatmap(pcn.pcmodule.layers[2].ps)
heatmap(pcn.initializer!.initializers[1].ps)
to_gpu!(pcn)







#initialize layers
l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2; prop_zero = 0.5, σ = relu, tc = 1.0f0, α = 0.05f0)

#initialize module and network
mo = PCModule(l0, (l1, l2))
pcn = PCNet(mo)
to_gpu!(pcn)#move to GPU 

x = cu(features[:, ind]) #move to GPU 
y = cu(labels[:, ind])

reset!(pcn)

########## Running the untrained network ##########

@time pcn(x, y; tspan = (0.0f0, 35.0f0), abstol = 0.01f0, reltol = 0.01f0, save_everystep = false);

obs = 10
yh = pcn.sol.u[end]
scatterlines(Array(yh.L1)[:, obs])

tspan = (0.0f0, 35.0f0)
 abstol = 0.01f0
 reltol = 0.01f0 
 save_everystep = false

@time odeprob = ODEProblem(pcn.odemodule, pcn.odemodule.u0, tspan, Float32[]);
@time integrator = init(odeprob, BS3(), abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = false);
odeprob.u0.L0 .= x



@time solve!(integrator);

lines(Array(integrator.sol.u[1].L1[:, 10]))
lines(Array(odeprob.u0.L0[:, 10]))

integrator.sol


########## Supervised Training ##########
reset!(pcn) #set all errors and predictions to 0
@time train!(pcn, x, y, (0.0f0, 30.0f0); iters = 1500, decayLrEvery = 50, lrDecayRate = 0.85f0, show_every = 10, normalize_every = 10000, abstol = 0.02f0, reltol = 0.05f0, stops = 25.0f0:1.0f0:30.0f0)

scatterlines(pcn.initerror)
scatterlines(pcn.error)


@time pcn(x, (0.0f0, 35.0f0), abstol = 0.01f0, reltol = 0.01f0);
pcn.sol.u[end].L2
guesses = Array(pcn.sol.u[end].L2)
labels = Array(y)

########## Evaluation ##########
      

pick_max!(guesses)
guesses

correct = dot(reshape(guesses, length(guesses)), reshape(labels, length(labels)))
total = sum(labels)
