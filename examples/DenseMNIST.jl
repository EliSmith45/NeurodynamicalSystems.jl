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


trainSize = 10000

trainFeatures, trainLabels = read_mnist_dense(trainSize, :train)

##### Initialize the network 
nObs = size(trainFeatures)[2]

# layer dimensions, must be a tuple
n0 = size(trainFeatures)# size of one sample
n1 = (128, nObs)
n2 = (64, nObs)
n3 = (32, nObs)
n4 = (10, nObs)

# create layers
l0 = PCStaticInput(n0, :L0);
l1 = PCDense(n1, n0, :L1; σ = relu, shrinkage = 0.1f0);
l2 = PCDense(n2, n1, :L2; σ = relu, shrinkage = 0.1f0);
l3 = PCDense(n3, n2, :L3; σ = relu, shrinkage = 0.1f0);
l4 = PCDense(n4, n3, :L4; σ = relu, shrinkage = 0.1f0);


mo = PCModule(l0, (l1, l2, l3, l4));


fSolver = ForwardEulerSolver(mo, dt = 0.05f0);
bSolver = BackwardEulerSolver(mo, dt = 0.01f0);
pcn = PCNetwork(mo, fSolver, bSolver);

to_gpu!(pcn);
trainingData = NamedTuple((L0 = cu(trainFeatures), L4 = cu(trainLabels)));
@time pcn(trainingData; maxIters = 50, stoppingCondition = 0.01f0, reinit = true)

scatterlines(get_du_logs(pcn)) 
scatterlines(get_error_logs(pcn))

GC.gc(true)


# train the network
@time train!(pcn, trainingData; maxIters = 50, stoppingCondition = 0.01f0, followUpRuns = 100000, maxFollowUpIters = 5, print_batch_error = 100)
change_step_size_backward!(pcn, (dt = 0.00015f0,))



scatterlines(get_training_du_logs(pcn)[10:100:end])
scatterlines(get_training_error_logs(pcn)[10:100:end])
# run the newly trained network
trainFeaturesc = cu(trainFeatures[:, 1:1024])
@time pcn(trainFeaturesc; maxIters = 100, stoppingCondition = 0.01f0, reinit = true)

scatterlines(get_du_logs(pcn))
scatterlines(get_error_logs(pcn))


