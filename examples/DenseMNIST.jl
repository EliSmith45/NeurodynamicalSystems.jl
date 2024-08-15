#using Pkg, CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
#Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;

using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA, MLDatasets, CairoMakie
using Flux: Flux, DataLoader


function read_mnist(trainsize = 10000, testsize = 1000)
    x = MNIST(Tx = Float32, split = :train) #load MNIST data
    xt = MNIST(Tx = Float32, split = :test) #load MNIST data
    features = reshape(x.features, size(x.features, 1) * size(x.features, 2), size(x.features, 3)) #add the third (channel) dimension
    featurest = reshape(xt.features, size(xt.features, 1) * size(xt.features, 2), size(xt.features, 3)) #add the third (channel) dimension
    
    
    #get the labels in a floating point one-hot matrix
    labels = zeros(eltype(x.features), length(unique(x.targets)), length(x.targets))
    labelst = zeros(eltype(xt.features), length(unique(xt.targets)), length(xt.targets))
    for i in eachindex(x.targets)
        labels[x.targets[i] + 1, i] = 1.0f0
    end
    
    for i in eachindex(xt.targets)
        labelst[xt.targets[i] + 1, i] = 1.0f0
    end
    
    
    #trainInd = StatsBase.sample(1:size(features, 2), trainSize, replace = false)
    #testInd = setdiff(1:size(features, 2), trainInd)
    #trainFeatures = features[:, trainInd] #add the third (channel) dimension
    #trainLabels = labels[:, trainInd] #add the third (channel) dimension
    return features[:, 1:trainsize], labels[:, 1:trainsize], featurest[:, 1:testsize], labelst[:, 1:testsize]
end

function classification_accuracy(pcn, data)
    sol = pcn(data.data.data, maxSteps = 50, stoppingCondition = 0.01f0, reset_module = true);
    dot(pick_max!(sol.L3), data.data.label) / size(data.data.label)[end]
end


trainSize = 1000
testSize = 100
trainFeatures, trainLabels, testFeatures, testLabels = read_mnist(trainSize, testSize)

##### Initialize the network 

#layer dimensions, must be a tuple
n0 = (size(trainFeatures, 1),)

n1 = (64,)
n2 = (64,)
n3 = (10,)

#initialize layers


l0 = PCStaticInput(n0, :L0)
l1 = PCDense2(n1, n0, relu, 0.1f0, false, :L1, Float32)
l2 = PCDense2(n2, n1, relu, 0.1f0, false, :L2, Float32)
mo = PCModule(l0, (l1, l2))
change_nObs!(mo, trainSize)

fSolver = forwardES1(mo, dt = 0.000001f0, maxSteps = 25)
bSolver = backwardES1(mo, dt = 0.000001f0, maxSteps = 25)

pcn = Pnet(mo, fSolver, bSolver)
x = ones(Float32, n0[1], 5000)

@time pcn(x)

@time backwardSolverStep!(pcn.psOpt)
@time trainSteps!(pcn, x; steps = 1, followUpRuns = 1, maxFollowUpSteps = 5)

to_gpu!(pcn)
xc = cu(x)

@time pcn(xc)

@time trainSteps!(pcn, xc; steps = 10, followUpRuns = 10, maxFollowUpSteps = 5)

to_cpu!(pcn)

heatmap(mo.ps.params.L2)

mo.u


x

asdf
####


x
























l0 = PCStaticInput((n0, trainSize), :L0)
l1, init1 = PCDense((n0, trainSize), (n1, trainSize), :L1; prop_zero = 0.25, σ = relu, tc = 1.0f0, α = 0.05f0, threshold = 0.1f0)
l2, init2 = PCDense((n1, trainSize), (n2, trainSize), :L2; prop_zero = 0.25, σ = relu, tc = 1.0f0, α = 0.05f0, threshold = 0.1f0)
l3, init3 = PCDense((n2, trainSize), (n3, trainSize), :L3; prop_zero = 0.0, σ = relu, tc = 1.0f0, α = 0.05f0, threshold = 0.1f0)

#initialize module and network
mo, initializer = PCModule(l0, (l1, l2, l3), (init1, init2, init3))
fps = ODEIntegrator(mo; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false, dt = 0.05f0, adaptive = true, dtmax = 1.0f0, dtmin = 0.0001f0)
pcn = PCNet(mo, initializer, fps)
to_gpu!(pcn)
reset!(pcn)

trainFeaturesc = cu(trainFeatures)
trainLabelsc = cu(trainLabels)
testFeaturesc = cu(testFeatures)
testLabelsc = cu(testLabels)

bs = 1024 
trainingData = DataLoader((data = trainFeaturesc, label = trainLabelsc), batchsize = bs, partial = false, shuffle = true)
testingData = DataLoader((data = testFeaturesc, label = testLabelsc), batchsize = bs, partial = false, shuffle = true)

GC.gc(true)


@time train!(pcn, trainingData; maxSteps = 50, stoppingCondition = 0.01, maxFollowupSteps = 10, epochs = 1000, trainstepsPerBatch = 5, decayLrEvery = 50, lrDecayRate = 0.9f0, show_every = 1, normalize_every = 1, trainingEvalCallback = (1, classification_accuracy), testingEvalCallback = (1, classification_accuracy), testData = testingData)



@time sol = pcn(trainFeaturesc[:, 1:1024], maxSteps = 50, stoppingCondition = 0.01f0, reset_module = true);
scatterlines(pcn.fixedPointSolver.errorLogs)
pcn.fixedPointSolver.u.L2
pick_max!(pcn.fixedPointSolver.u.L2)
trainLabelsc[:, 1:1024]
aa = dot(pcn.fixedPointSolver.u.L2, trainLabelsc[:, 1:1024]) / 1024

@time sol = pcn(trainingData.data.data, maxSteps = 50, stoppingCondition = 0.01f0, reset_module = true);

########## Evaluation ##########

classification_accuracy(pcn, trainingData)



pick_max!(guesses)
guesses

correct = dot(reshape(guesses, length(guesses)), reshape(labels, length(labels)))
total = sum(labels)
