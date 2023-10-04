using Pkg, CairoMakie
cd("NeurodynamicalSystems"); #navigate to the package directory
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;

using StatsBase, LinearAlgebra, ComponentArrays, DifferentialEquations, CUDA, MLDatasets, NNlib, NNlibCUDA



#x = MNIST(Tx = Float32, split = :train)
nObs = 256
sample(1:10, 2)
#set up the network

x = rand(Float32, 28, 28, 1, nObs)

l0 = PCInput((28, 28, 1, nObs), :L0);
l1 = PCConv((3, 3), (1 => 8), (28, 28, 1, nObs), :L1, σ = relu, α = 0.001f0);
l2 = PCConv((3, 3), (8 => 32), (26, 26, 8, nObs), :L2, σ = relu, α = 0.001f0);
l3 = PCConv((3, 3), (32 => 64), (24, 24, 32, nObs), :L3, σ = relu, α = 0.001f0);

#initialize module and network
mo = CompositeModule(l0, (l1, l2));
pcn = PCNet(mo);
@time pcn(x, (0.0f0, 10.0f0), abstol = 0.015f0, reltol = 0.1f0);
########## Running the network on the input x ##########
to_gpu!(pcn)
xc = cu(x)
l1.initializer!.ps .*= 0.0f0
l2.initializer!.ps .*= 0.0f0
l3.initializer!.ps .*= 0.0f0
@time pcn(xc, (0.0f0, 10.0f0), abstol = 0.015f0, reltol = 0.1f0);

reset!(pcn)

