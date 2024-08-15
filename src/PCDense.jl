

module PCDenseLayers2


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./CommonFunctions.jl")

@reexport using .CommonFunctions

export PCDense2, PCStaticInput  #, change_nObs!

mutable struct PCDense2

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons has a state size of (64,)
    inputsize
    
    σ #activation function

    
    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    is_supervised #whether the layer is used for supervised learning
    name
    T
   
end


function CommonFunctions.allocate_params(layer::PCDense2)
    ps = relu(rand(layer.T, layer.inputsize..., layer.statesize...) .- 0.5f0)
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    #ps
end
function CommonFunctions.allocate_initparams(layer::PCDense2)
    ps = rand(layer.T, layer.statesize..., layer.inputsize...) .- 0.5f0
end

function CommonFunctions.allocate_receptive_field_norms(layer::PCDense2)
    zeros(layer.T, 1, layer.statesize...)
end

function CommonFunctions.make_predictions!(predictions_lm1, layer::PCDense2, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
    return
end


function CommonFunctions.get_gradient_activations!(du, layer::PCDense2, errors_lm1, errors_l, ps)
    du .= mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.threshold
end

function CommonFunctions.get_gradient_parameters!(grads, layer::PCDense2, errors_lm1, u_l)
    mul!(grads, errors_lm1, transpose(u_l))
end

function CommonFunctions.get_gradient_init_parameters!(grads, layer::PCDense2, errors_l, u_lm1)
    mul!(grads, errors_l, transpose(u_lm1))
end

function CommonFunctions.get_u0!(u0, layer::PCDense2, initps, x)
    u0 .= layer.σ.(mul!(u0, initps, x))
end



##### Input layers #####

"""
    PCStaticInput(in_dims, states, name, T)
Input layer who holds the data and does not change over time.
"""
mutable struct PCStaticInput

    statesize
    input 
    name
    T

end

"""
    PCStaticInput(in_dims::Tuple, name::Symbol, T = Float32)
Constructor for PCDynamicInput ayers.
"""
function PCStaticInput(in_dims, name::Symbol, T = Float32)
    input = zeros(T, in_dims..., 1)
    PCStaticInput(in_dims, input, name, T)
end

# Makes the static input layer callable to calculate state updates in the ODE system
function CommonFunctions.get_gradient_activations!(du, u, layer::PCStaticInput, errors_l)
    du .= 0
    u .= layer.input
end






end