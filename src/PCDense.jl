

module PCDenseLayers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./CommonFunctions.jl")

@reexport import .CommonFunctions

export PCDense  #, change_nObs!

mutable struct PCDense

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons has a state size of (64,)
    inputsize
    
    σ #activation function

    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end


function CommonFunctions.allocate_params(layer::PCDense)
    ps = relu(rand(layer.T, layer.inputsize..., layer.statesize...) .- 0.5f0)
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    #ps
end
function CommonFunctions.allocate_initparams(layer::PCDense)
    ps = rand(layer.T, layer.statesize..., layer.inputsize...) .- 0.5f0
end

function CommonFunctions.allocate_receptive_field_norms(layer::PCDense)
    zeros(layer.T, 1, layer.statesize...)
end

function CommonFunctions.make_predictions!(predictions_lm1, layer::PCDense, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
    return
end


function CommonFunctions.get_gradient_activations!(du, layer::PCDense, errors_lm1, errors_l, ps)
    du .= mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.threshold
end

function CommonFunctions.get_gradient_parameters!(grads, layer::PCDense, errors_lm1, u_l)
    mul!(grads, errors_lm1, transpose(u_l))
end

function CommonFunctions.get_gradient_init_parameters!(grads, layer::PCDense, errors_l, u_lm1)
    mul!(grads, errors_l, transpose(u_lm1))
end

function CommonFunctions.get_u0!(u0, layer::PCDense, initps, x)
    u0 .= layer.σ.(mul!(u0, initps, x))
end


end