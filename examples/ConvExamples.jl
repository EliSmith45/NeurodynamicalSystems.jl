using Revise;
using NeurodynamicalSystems;



using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, CairoMakie, MLDatasets
using Flux: Flux, DataLoader #, Conv, ConvTranspose, conv_transpose_dims


##### data preprocessing #####

# preprocess within a function to avoid polluting the global namespace and heap memory
function read_mnist_conv(trainsize = 10000, split = :train)
    x = MNIST(Tx = Float32, split = split) #load MNIST data
    features = reshape(x.features, size(x.features, 1), size(x.features, 2), 1, size(x.features, 3))
    
    #get the labels in a floating point one-hot matrix
    labels = zeros(eltype(x.features), length(unique(x.targets)), length(x.targets))
    for i in eachindex(x.targets)
        labels[x.targets[i] + 1, i] = 1.0f0
    end
    
    #ind = StatsBase.sample(1:size(features)[end], trainsize, replace = false)
    return features[:, :, :, 1:trainsize], labels[:, 1:trainsize]
end


features, labels = read_mnist_conv(60000, :train)
##### examples of convolutional predictive coding networks on MNIST #####
heatmap(features[:, :, 1, 1]')

labels[:, 1]
#set up the network
nObs = 12
l0 = PCStaticInput((28, 28, 1, nObs), :L0);
l1 = PCConv((5, 5), (1 => 16), get_state_size(l0), :L1; σ = relu, padding = (2, 2), shrinkage = .1f0);
l2 = PCConv2Dense((10, nObs), get_state_size(l1), :L2; σ = relu, shrinkage = .1f0);

mo = PCModule(l0, (l1, l2));
fSolver = ForwardEulerSolver(mo, dt = 0.025f0);
bSolver = BackwardEulerSolver(mo, dt = 0.000005f0);
pcn = PCNetwork(mo, fSolver, bSolver);

# run the network on the CPU if you're curious about how much faster the GPU is
#=
@time pcn(xd; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
=#


featuresc = cu(features[:, :, :, 1:12]) #start small during compilation!
labelsc = cu(labels[:, 1:12])
to_gpu!(pcn)
@time pcn(featuresc; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = true, reset_states = true) 
@time pcn(featuresc, labelsc; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = true, reset_states = true) 

scatterlines(get_du_logs(pcn)) #make sure the forward pass step size is acceptable
scatterlines(get_error_logs(pcn))

change_step_size_forward!(pcn, (dt = .01f0,))

@time pcn(featuresc, labelsc; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = true, reset_states = true) 
scatterlines(get_du_logs(pcn)) #make sure the forward pass step size is acceptable
scatterlines(get_error_logs(pcn))

get_states(pcn).L2

#train the network

recommend_batch_size(pcn, 5.0)
batchSize = 1024

nObs = 24000
trainingData = DataLoader((data = features[:, :, :, 1:nObs], label = labels[:, 1:nObs]), batchsize = batchSize, partial = false, shuffle = true)
@time train_supervised!(pcn, trainingData; maxIters = 50, stoppingCondition = 0.01f0, epochs = 5000, followUpRuns = 200, maxFollowUpIters = 5)


change_step_size_backward!(pcn, (dt = .0000015f0,))


scatterlines(get_training_du_logs(pcn))
scatterlines(get_training_error_logs(pcn)[10:end])



featuresc = cu(features[:, :, :, 1:12]) #start small during compilation!
labelsc = cu(labels[:, 1:12])

@time pcn(featuresc; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = true, reset_states = true) 
get_states(pcn).L5
labelsc


#change_step_size_forward!(pcn, (dt = .02f0,))
@time pcn(featuresc; maxIters = 100, stoppingCondition = 0.01f0, use_neural_initializer = true, reset_states = true) 
scatterlines(get_du_logs(pcn)) #make sure the forward pass step size is acceptable
scatterlines(get_error_logs(pcn))

get_states(pcn).L5

labelsc

########## Running the network on the input x ##########
to_gpu!(pcn)
xc = cu(xd)
#l1.initializer!.ps .*= 0.0f0
#l2.initializer!.ps .*= 0.0f0
#l3.initializer!.ps .*= 0.0f0
@time pcn(xc, (0.0f0, 10.0f0), abstol = 0.015f0, reltol = 0.1f0);
GC.gc()
reset!(pcn)

