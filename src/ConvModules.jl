module ConvModules

#= 
Data structures and functions for convolutional modules. These are used when the input is a structured array like an image or frequency spectrum.
Scale levels are connected via convolutions.
=#

########## External Dependencies ##########
using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, CUDA


########## Internal Dependencies ##########
include("./Utils.jl")
using .Utils

########## Exports ##########
export PCConv, ConvModule, ConvInitializer

########## Data structures ##########

# Convolutional layer. This is not callable as in Flux layers, but is used as a building block for ConvModules.
mutable struct PCConv

    states
    ps
    grads
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    cdims
    tc
    α
    name

end



# Convolutional module. This is a callable structure that stores and updates convolutionally connected layers in the ODE solver.
mutable struct ConvModule

    is_supervised
    inputstates
    labels
    ps
    grads
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    cdims
    constants
    u0
    predictions
    errors
    initializer!

end



# Callable struct that initializes u0 before running the ODE solver with a feedforward convolutionally connected network with ReLU activations.
mutable struct ConvInitializer

    errors
    ps
    grads
    cdims
    α

end


########## Constructors ##########

# Construct a convolutional hidden layer.
function PCConv(k, ch::Pair, in_dims, name::Symbol, T = Float32; stride=1, padding=0, dilation=1, groups=1, tc = 0.1f0, α = 0.01f0)

    inputstates = zeros(T, in_dims...)
    ps = rand(T, k..., ch[1], ch[2])
    broadcast!(relu, ps, ps)

    grads = deepcopy(ps)
    ps2 = deepcopy(ps) .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1:(ndims(ps) - 1))
    ps ./= receptiveFieldNorms

    cdims = NNlib.DenseConvDims(size(inputstates), size(ps); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    states = NNlib.conv(inputstates, ps, cdims)
    
   
    PCConv(states, ps, grads, ps2, receptiveFieldNorms, cdims, tc, α, name)
end


# Construct a convolutional module from at least three convolutional layers. 
function ConvModule(inputlayer, hiddenlayers, toplayer, is_supervised = false)
    
    inputstates = inputlayer.states
    labels = toplayer.states

    layers = (inputlayer, hiddenlayers..., toplayer)

    names = map(l -> l.name, layers)
    ps = map(l -> l.ps, layers[2:end])
    grads =  map(l -> l.grads, layers[2:end])
    ps2 = map(l -> l.ps2, layers[2:end])
    receptiveFieldNorms = map(l -> l.receptiveFieldNorms, layers[2:end])
    tc = [map(l -> l.tc, layers[2:end])...]
    α = [map(l -> l.α, layers[2:end])...]
    
    errors = NamedTuple{names}(map(l -> deepcopy(l.states), layers))
    predictions = NamedTuple{names}(map(l -> deepcopy(l.states), layers))
    u0 = NamedTuple{names}(map(l -> deepcopy(l.states), layers))

    errors = ComponentArray(errors)
    predictions = ComponentArray(predictions)
    u0 = ComponentArray(u0)


    initializer! = ConvInitializer(deepcopy(errors), deepcopy(ps), deepcopy(grads), cdims, α)
    
    ConvModule(is_supervised, inputstates, labels, ps, grads, ps2, receptiveFieldNorms, tc, α, u0, predictions, errors, initializer!)

end



########## Functions ##########

# Makes the convolutional module callable to compute the activation updates within the ODE solver. 
function (m::ConvModule)(du, u, p, t)
    
    broadcast!(relu, u, u)
    values(NamedTuple(u))[1] .= m.inputstates
    values(NamedTuple(du))[1] .= zero(eltype(du))
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(u))[end] 

    if m.is_supervised
        values(NamedTuple(u))[end] .= m.labels
    end
    
    m.errors .= u .- m.predictions


    for k in eachindex(m.tc)
        NNlib.conv!(values(NamedTuple(m.predictions))[k], values(NamedTuple(u))[k + 1], m.ps[k], m.cdims[k])
        NNlib.∇conv_data!(values(NamedTuple(du))[k + 1], values(NamedTuple(m.errors))[k], m.ps[k], m.cdims[k])
       
        values(NamedTuple(du))[k + 1] .-= values(NamedTuple(m.errors))[k + 1]
        values(NamedTuple(du))[k + 1] .*= m.tc[k]
    end
    
   
end


# Makes the ConvInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::ConvInitializer)(u0, x)
    u0 .= 0.0f0
    values(NamedTuple(u0))[1] .= x
    for k in eachindex(m.ps)
        NNlib.∇conv_data!(values(NamedTuple(u0))[k + 1], values(NamedTuple(u0))[k], m.ps[k], m.cdims[k])
        broadcast!(relu, values(NamedTuple(u0))[k + 1], values(NamedTuple(u0))[k + 1])
    end
end

# Makes PCModule callable on the integrator object of the ODE solver. This function updates
# each layer's parameters via a callback function.
function (m::ConvModule)(integrator)
    
    for k in eachindex(m.tc)
       
        NNlib.∇conv_filter!(m.grads[k], values(NamedTuple(m.errors))[k],  values(NamedTuple(integrator.u))[k + 1])
        m.ps[k] .+= m.constants.α[k] .* m.grads[k]
        #nonneg_normalized!(@view(m.ps[lp1]))
    end

end


end