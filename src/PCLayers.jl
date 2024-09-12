module PCLayers

# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport


export PCDense, PCStaticInput, PCConv, PCConv2Dense, PCLayer
export allocate_states, allocate_params, allocate_initparams, allocate_receptive_field_norms, change_nObs!, get_state_size, make_predictions!, get_gradient_activations!, get_gradient_parameters!, get_gradient_init_parameters!, get_u0!

##### Abstract predictive coding layer type #####

#=
These functions have the same implementation for most layer types, but can be overloaded if a specific layer requires
these functions to have other side effects.
=#

abstract type PCLayer end

function allocate_states(layer::PCLayer)
    zeros(layer.T, layer.statesize...)
end
    
function allocate_params(layer::PCLayer)

    ps = relu(rand(layer.T, layer.psSize) .- 0.5f0)
    rfnorms = sqrt.(sum(ps .^2, dims = 1:(ndims(ps) - 1)))
    ps ./= rfnorms
    ps

end

function allocate_initparams(layer::PCLayer)
    
    rand(layer.T, layer.initpsSize) .- 0.5f0
    ps = rand(layer.T, layer.initpsSize) .- 0.5f0
    rfnorms = sqrt.(sum(ps .^2, dims = 1:(ndims(ps) - 1)))
    ps ./= rfnorms
    ps

end

function allocate_receptive_field_norms(layer::PCLayer)
    ps = zeros(layer.T, layer.psSize) 
    sum(ps, dims = 1:(ndims(ps) - 1))
end

function change_nObs!(layer::PCLayer, nObs)
    layer.statesize = [layer.statesize...]
    layer.inputsize = [layer.inputsize...]
    layer.statesize[end] = nObs
    layer.inputsize[end] = nObs
    layer.statesize = Tuple(layer.statesize)
    layer.inputsize = Tuple(layer.inputsize)
end

function get_state_size(layer::PCLayer)
    layer.statesize
end
##### Dense layers #####

mutable struct PCDense <: PCLayer

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons observing 1 sample has a state size of (64, 1)
    inputsize
    psSize
    initpsSize
    
    σ #activation function, probably should always be relu

    shrinkage #minimum activation. Each iteration, the shrinkage is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end


"""
    PCDense(statesize, inputsize, name::Symbol, T = Float32; σ = relu, shrinkage = 0.05f0)

Constructs a dense layer for a predictive coding network. That is, the states of this layer are transformed by dense matrix to predict the layer below.

# Arguments
- `statesize`: A 2-tuple specifying the size of the state vector. The first element is the number of neurons in the layer, the second element is the number of observations.
- `inputsize`: A 2-tuple specifying the size of the input vector. The first element is the number of neurons in the layer below, the second element is the number of observations.
- `name::Symbol`: The name of the layer.
- `T`: The data type to be used for calculations (default: Float32).
- `σ`: The activation function to be used (default: relu).
- `shrinkage`: The shrinkage value. With each forward pass step, this value is subtracted from the state. This encourages sparsity, accelerates convergence, and sets the minimum possible activation (default: 0.05f0).

# Returns
- An instance of the PCDense layer.

# Examples

"""

function PCDense(statesize, inputsize, name::Symbol, T = Float32; σ = relu, shrinkage = 0.05f0)

    psSize = Tuple([inputsize[1], statesize[1]])
    initpsSize = Tuple([statesize[1], inputsize[1]])

    PCDense(statesize, inputsize, psSize, initpsSize, σ, shrinkage, name, T)

end

function make_predictions!(predictions_lm1, layer::PCDense, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
    return
end


function get_gradient_activations!(du, layer::PCDense, errors_lm1, errors_l, ps)
    du .= mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.shrinkage
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


##### Convolutional layers #####


mutable struct PCConv <: PCLayer

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons has a state size of (64,)
    inputsize
    psSize
    initpsSize

    stride
    padding
    dilation
    groups
    flipkernel

    cdims
    initcdims
    σ #activation function

    
    shrinkage #minimum activation. Each iteration, the shrinkage is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end


"""
    PCConv(k, ch::Pair, inputsize, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, flipkern = true, shrinkage = 0.05f0)

Constructs a PCConv layer. The layer below is predicted by applying a convolution to the state, and the states are updated by a convolution transpose.

# Arguments
- `k`: Kernel size.
- `ch::Pair`: Number of input and output channels.
- `inputsize`: Size of the input.
- `name::Symbol`: Name of the layer.
- `T`: Data type (default: Float32).
- `prop_zero`: Proportion of zero values in the kernel (default: 0.5).
- `σ`: Activation function (default: relu).
- `stride`: Stride size (default: 1).
- `padding`: Padding size (default: 0).
- `dilation`: Dilation size (default: 1).
- `groups`: Number of groups (default: 1).
- `flipkern`: Whether to flip the kernel (default: true).
- `shrinkage`: Shrinkage factor (default: 0.05f0).
"""

function PCConv(k, ch::Pair, inputsize, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, flipkern = true, shrinkage = 0.05f0)

    psSize = Tuple([k..., ch[2], ch[1]])
    initpsSize = Tuple([k..., ch[1], ch[2]])

    initcdims = NNlib.DenseConvDims(inputsize, initpsSize; stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = flipkern)
    y = NNlib.conv(rand(T, inputsize), rand(T, initpsSize), initcdims)
    statesize = size(y)


    cdims = NNlib.DenseConvDims(statesize, psSize; stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = flipkern)
    initcdims = NNlib.DenseConvDims(inputsize, initpsSize; stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = flipkern)
    PCConv(statesize, inputsize, psSize, initpsSize, stride, padding, dilation, groups, flipkern, cdims, initcdims, σ, shrinkage, name, T)

end

#=
function allocate_states(layer::PCConv)
    zeros(layer.T, layer.statesize...)
end
    
function allocate_params(layer::PCConv)
    ps = relu(rand(layer.T, layer.cdims.kernel_size..., layer.cdims.channels_in, layer.cdims.channels_out) .- 0.5f0)
end
function allocate_initparams(layer::PCConv)
    ps = rand(layer.T, layer.initcdims.kernel_size..., layer.initcdims.channels_in, layer.initcdims.channels_out) .- 0.5f0
end
=#


function allocate_states(layer::PCConv)
    zeros(layer.T, layer.statesize...)
end
    
function allocate_params(layer::PCConv)

    ps = relu(rand(layer.T, layer.psSize) .- 0.5f0)
    rfnorms = sqrt.(sum(ps .^2, dims = [1:(ndims(ps) - 2)..., ndims(ps)]))
    ps ./= rfnorms
    ps

end

function allocate_initparams(layer::PCConv)
    
    rand(layer.T, layer.initpsSize) .- 0.5f0
    ps = rand(layer.T, layer.initpsSize) .- 0.5f0
    rfnorms = sqrt.(sum(ps .^2, dims = 1:(ndims(ps) - 1)))
    ps ./= rfnorms
    ps

end

function allocate_receptive_field_norms(layer::PCConv)
    ps = zeros(layer.T, layer.psSize) 
    sum(ps, dims = [1:(ndims(ps) - 2)..., ndims(ps)])
end


function change_nObs!(layer::PCConv, nObs)
    layer.statesize = [layer.statesize...]
    layer.inputsize = [layer.inputsize...]
    layer.statesize[end] = nObs
    layer.inputsize[end] = nObs
    layer.statesize = Tuple(layer.statesize)
    layer.inputsize = Tuple(layer.inputsize)

    
    layer.cdims = NNlib.DenseConvDims(layer.statesize, layer.psSize; stride = layer.stride, padding = layer.padding, dilation = layer.dilation, groups = layer.groups, flipkernel = layer.flipkernel)
    layer.initcdims = NNlib.DenseConvDims(layer.inputsize, layer.initpsSize; stride = layer.stride, padding = layer.padding, dilation = layer.dilation, groups = layer.groups, flipkernel = layer.flipkernel)
   
end

function make_predictions!(predictions_lm1, layer::PCConv, ps, u_l)
    conv!(predictions_lm1, u_l, ps, layer.cdims)
    return
end


function get_gradient_activations!(du, layer::PCConv, errors_lm1, errors_l, ps)
    du .= ∇conv_data!(du, errors_lm1, ps, layer.cdims) .- errors_l .- layer.shrinkage
end

function get_gradient_parameters!(grads, layer::PCConv, errors_lm1, u_l)
    ∇conv_filter!(grads, u_l, errors_lm1, layer.cdims)
end

function get_gradient_init_parameters!(grads, layer::PCConv, errors_l, u_lm1)
    ∇conv_filter!(grads, u_lm1, errors_l, layer.initcdims)
end

function get_u0!(u0, layer::PCConv, initps, x)
    conv!(u0, x, initps, layer.initcdims)
end


##### Conv2Dense layers #####


mutable struct PCConv2Dense <: PCLayer

    statesize #size of state array
    inputsize
    inputsizeFlat
    psSize
    initpsSize
    σ #activation function

    predictedFlat
    errorsFlat
    
    shrinkage #minimum activation. Each iteration, the shrinkage is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end


"""
    PCConv2Dense(statesize, inputsize, name::Symbol, T = Float32; σ = relu, shrinkage = 0.1f0)

Constructs a PCConv2Dense layer.

# Arguments
- `statesize::Tuple`: The size of the output state.
- `inputsize::Tuple`: The size of the input.
- `name::Symbol`: The name of the layer.
- `T::Type`: The type of the layer's parameters. Default is `Float32`.
- `σ::Function`: The activation function. Default is `relu`.
- `shrinkage::Float`: The shrinkage factor. Default is `0.1f0`.

# Returns
- `PCConv2Dense`: The constructed PCConv2Dense layer.
"""
function PCConv2Dense(statesize, inputsize, name::Symbol, T = Float32; σ = relu, shrinkage = 0.1f0)

    inputsizeFlat = (prod(inputsize[1:end-1]), inputsize[end])
    predictedFlat = zeros(T, inputsizeFlat)
    errorsFlat = zeros(T, inputsizeFlat)

    psSize = Tuple([inputsizeFlat[1], statesize[1]])
    initpsSize = Tuple([statesize[1], inputsizeFlat[1]])

    PCConv2Dense(statesize, inputsize, inputsizeFlat, psSize, initpsSize, σ, predictedFlat, errorsFlat, shrinkage, name, T)

end

function allocate_states(layer::PCConv2Dense)
    layer.predictedFlat = zeros(layer.T, layer.inputsizeFlat)
    layer.errorsFlat = zeros(layer.T, layer.inputsizeFlat)
    zeros(layer.T, layer.statesize...)
end
    

function change_nObs!(layer::PCConv2Dense, nObs)

    layer.statesize = [layer.statesize...]
    layer.inputsize = [layer.inputsize...]
    layer.inputsizeFlat = [layer.inputsizeFlat...]
    layer.statesize[end] = nObs
    layer.inputsize[end] = nObs
    layer.inputsizeFlat[end] = nObs
    layer.statesize = Tuple(layer.statesize)
    layer.inputsize = Tuple(layer.inputsize)
    layer.inputsizeFlat = Tuple(layer.inputsizeFlat)

end


function make_predictions!(predictions_lm1, layer::PCConv2Dense, ps, u_l)
    predictions_lm1 .= reshape(mul!(layer.predictedFlat, ps, u_l), layer.inputsize)
    return
end


function get_gradient_activations!(du, layer::PCConv2Dense, errors_lm1, errors_l, ps)
    layer.errorsFlat .= reshape(errors_lm1, layer.inputsizeFlat)
    du .= mul!(du, transpose(ps), layer.errorsFlat) .- errors_l .- layer.shrinkage
end

function get_gradient_parameters!(grads, layer::PCConv2Dense, errors_lm1, u_l)
    layer.errorsFlat .= reshape(errors_lm1, layer.inputsizeFlat)
    mul!(grads, layer.errorsFlat, transpose(u_l))
end

function get_gradient_init_parameters!(grads, layer::PCConv2Dense, errors_l, u_lm1)
    layer.predictedFlat .= reshape(u_lm1, layer.inputsizeFlat) #u_lm1 is the input to the layer, we use predictedFlat to store the reshaped input to save memory
    mul!(grads, errors_l, transpose(layer.predictedFlat))
end

function get_u0!(u0, layer::PCConv2Dense, initps, x)
    layer.predictedFlat .= reshape(x, layer.inputsizeFlat) #x is the input to the layer, we use predictedFlat to store the reshaped input to save memory
    u0 .= layer.σ.(mul!(u0, initps, layer.predictedFlat ))
end



##### Input layers #####

"""
    PCStaticInput(in_dims, states, name, T)
Input layer who holds the data and does not change over time.
"""
mutable struct PCStaticInput <: PCLayer

    statesize
    inputsize

    data 
    name
    T

end


"""
    PCStaticInput(in_dims, name::Symbol, T = Float32)

Create a PCStaticInput layer.

# Arguments
- `in_dims`: The input dimensions of the layer.
- `name::Symbol`: The name of the layer.
- `T`: The data type of the layer (default: Float32).

# Returns
- `PCStaticInput`: The PCStaticInput layer.

# Example

"""
function PCStaticInput(in_dims, name::Symbol, T = Float32)
    data = zeros(T, in_dims)
    PCStaticInput(in_dims, in_dims, data, name, T)
end

function allocate_states(layer::PCStaticInput)
    layer.data = zeros(layer.T, layer.statesize...)
    deepcopy(layer.data)
end

# Makes the static input layer callable to calculate state updates in the ODE system
function get_gradient_activations!(du, u, layer::PCStaticInput, errors_l)
    du .= 0
    u .= layer.data
end




end