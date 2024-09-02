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

trainSize = 60000
testSize = 1024
trainFeatures, trainLabels, testFeatures, testLabels = read_mnist(trainSize, testSize)

##### Initialize the network 
nObs = size(trainFeatures)[2]

# layer dimensions, must be a tuple
n0 = size(trainFeatures)# size of one sample
n1 = (64, nObs)
n2 = (10, nObs)

# create layers
l0 = PCStaticInput(n0, :L0)
l1 = PCDense(n1, n0, relu, 0.1f0, :L1, Float32)
l2 = PCDense(n2, n1, relu, 0.1f0, :L2, Float32)
mo = PCModule(l0, (l1, l2))


fSolver = ForwardEulerSolver(mo, dt = 0.02f0)
bSolver = BackwardEulerSolver(mo, dt = 0.01f0)
pcn = Pnet(mo, fSolver, bSolver)

to_gpu!(pcn)
trainFeaturesc = cu(trainFeatures)
trainLabelsc = cu(trainLabels)
testFeaturesc = cu(testFeatures)
testLabelsc = cu(testLabels)

bs = 1024 * 4
trainingData = DataLoader((data = trainFeaturesc, label = trainLabelsc), batchsize = bs, partial = false, shuffle = true)
testingData = DataLoader((data = testFeaturesc, label = testLabelsc), batchsize = bs, partial = false, shuffle = true)

GC.gc(true)


# train the network
@time trainSteps!(pcn, trainingData; maxIters = 50, stoppingCondition = 0.01f0, trainingSteps = 500, followUpRuns = 10, maxFollowUpIters = 5)

# look at the convergence of the training algorithm
scatterlines(get_training_du_logs(pcn))
scatterlines(get_training_error_logs(pcn))


# run the newly trained network
@time pcn(trainFeaturesc[:, 1:1024]; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))
pick_max!(values(NamedTuple(get_states(pcn)))[end])
dot(values(NamedTuple(get_states(pcn)))[end], trainLabelsc[:, 1:1024])

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
