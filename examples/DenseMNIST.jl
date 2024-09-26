#using Pkg, CairoMakie
#cd("NeurodynamicalSystems"); #navigate to the package directory
#Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;

using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA, MLDatasets, CairoMakie
using Flux: Flux, DataLoader


function read_mnist_dense(trainsize = 10000, split = :train)
    x = MNIST(Tx = Float32, split = split) #load MNIST data
    features = reshape(x.features, size(x.features, 1) * size(x.features, 2), size(x.features, 3)) #add the third (channel) dimension
  
    #get the labels in a floating point one-hot matrix
    labels = zeros(eltype(x.features), length(unique(x.targets)), length(x.targets))
    for i in eachindex(x.targets)
        labels[x.targets[i] + 1, i] = 1.0f0
    end
    
   
    
    
    #trainInd = StatsBase.sample(1:size(features, 2), trainSize, replace = false)
    #testInd = setdiff(1:size(features, 2), trainInd)
    #trainFeatures = features[:, trainInd] #add the third (channel) dimension
    #trainLabels = labels[:, trainInd] #add the third (channel) dimension
    return features[:, 1:trainsize], labels[:, 1:trainsize]
end


trainSize = 60000

trainFeatures, trainLabels = read_mnist_dense(trainSize, :train)

##### Initialize the network 
nObs = size(trainFeatures)[2]

# layer dimensions, must be a tuple
n0 = size(trainFeatures)# size of one sample
n1 = (128, nObs)
n2 = (128, nObs)
n3 = (128, nObs)
n4 = (128, nObs)
n5 = (128, nObs)
n6 = (64, nObs)
n7 = (32, nObs)
n8 = (10, nObs)

# create layers
l0 = PCStaticInput(n0, :L0);
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.1f0);
l2 = PCDense(n2, n1, :L2; σ = relu, shrinkage = 0.1f0);
l3 = PCDense(n3, n2, :L3; σ = relu, shrinkage = 0.1f0);
l4 = PCDense(n4, n3, :L4; σ = relu, shrinkage = 0.1f0);
l5 = PCDense(n5, n4, :L5; σ = relu, shrinkage = 0.1f0);
l6 = PCDense(n6, n5, :L6; σ = relu, shrinkage = 0.1f0);
l7 = PCDense(n7, n6, :L7; σ = relu, shrinkage = 0.1f0);
l8 = PCDense(n8, n7, :L8; σ = relu, shrinkage = 0.1f0);

mo = PCModule(l0, (l1, l2, l3, l4, l5, l6, l7, l8));


fSolver = ForwardEulerSolver(mo, dt = 0.025f0);
bSolver = BackwardEulerSolver(mo, dt = 0.0051f0);
pcn = PCNetwork(mo, fSolver, bSolver);

to_gpu!(pcn);
#trainFeaturesc = cu(trainFeatures)
#trainLabelsc = cu(trainLabels)
#testFeaturesc = cu(testFeatures)
#testLabelsc = cu(testLabels)

bs = 1024 * 2
trainingData = DataLoader((data = trainFeatures, label = trainLabels), batchsize = bs, partial = false, shuffle = true)

GC.gc(true)


# train the network
@time train_supervised!(pcn, trainingData; maxIters = 50, stoppingCondition = 0.01f0, epochs = 3, followUpRuns = 1, maxFollowUpIters = 5)
change_step_size_backward!(pcn, (dt = 0.00125f0,))

scatterlines(get_training_du_logs(pcn)[1000:1000:end])
scatterlines(get_training_error_logs(pcn)[15000:1000:end])
# run the newly trained network
trainFeaturesc = cu(trainFeatures[:, 1:1024])
@time pcn(trainFeaturesc; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))


