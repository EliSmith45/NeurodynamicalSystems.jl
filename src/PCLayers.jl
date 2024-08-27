module PCLayers

# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport


export PCDense, PCStaticInput
export allocate_states, allocate_params, allocate_initparams, allocate_receptive_field_norms, change_nObs!, make_predictions!, get_gradient_activations!, get_gradient_parameters!, get_gradient_init_parameters!, get_u0!

mutable struct PCDense

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons observing 1 sample has a state size of (64, 1)
    inputsize
    
    σ #activation function, probably should always be relu

    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end

function allocate_states(layer::PCDense)
    zeros(layer.T, layer.statesize...)
end
    
function allocate_params(layer::PCDense)
    ps = relu(rand(layer.T, layer.inputsize[1], layer.statesize[1]) .- 0.5f0)
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    #ps
end
function allocate_initparams(layer::PCDense)
    ps = rand(layer.T, layer.statesize[1], layer.inputsize[1]) .- 0.5f0
end

function allocate_receptive_field_norms(layer::PCDense)
    zeros(layer.T, 1, layer.statesize[1])
end

function change_nObs!(layer::PCDense, nObs)
    layer.statesize = [layer.statesize...]
    layer.inputsize = [layer.inputsize...]
    layer.statesize[end] = nObs
    layer.inputsize[end] = nObs
    layer.statesize = Tuple(layer.statesize)
    layer.inputsize = Tuple(layer.inputsize)
end
function make_predictions!(predictions_lm1, layer::PCDense, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
    return
end


function get_gradient_activations!(du, layer::PCDense, errors_lm1, errors_l, ps)
    du .= mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.threshold
end

function get_gradient_parameters!(grads, layer::PCDense, errors_lm1, u_l)
    mul!(grads, errors_lm1, transpose(u_l))
end

function get_gradient_init_parameters!(grads, layer::PCDense, errors_l, u_lm1)
    mul!(grads, errors_l, transpose(u_lm1))
end

function get_u0!(u0, layer::PCDense, initps, x)
    u0 .= layer.σ.(mul!(u0, initps, x))
end




##### Input layers #####

"""
    PCStaticInput(in_dims, states, name, T)
Input layer who holds the data and does not change over time.
"""
mutable struct PCStaticInput

    statesize
    inputsize

    data 
    name
    T

end

"""
    PCStaticInput(in_dims::Tuple, name::Symbol, T = Float32)
Constructor for PCDynamicInput ayers.
"""
function PCStaticInput(in_dims, name::Symbol, T = Float32)
    data = zeros(T, in_dims)
    PCStaticInput(in_dims, in_dims, data, name, T)
end

function allocate_states(layer::PCStaticInput)
    layer.data = zeros(layer.T, layer.statesize...)
    deepcopy(layer.data)
end

function change_nObs!(layer::PCStaticInput, nObs)
    layer.statesize = [layer.statesize...]
    layer.inputsize = [layer.inputsize...]
    layer.statesize[end] = nObs
    layer.inputsize[end] = nObs
    layer.statesize = Tuple(layer.statesize)
    layer.inputsize = Tuple(layer.inputsize)
end

# Makes the static input layer callable to calculate state updates in the ODE system
function get_gradient_activations!(du, u, layer::PCStaticInput, errors_l)
    du .= 0
    u .= layer.data
end




end