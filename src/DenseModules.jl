module DenseModules

#= 
Data structures and functions for dense modules. These are used when one input sample is a vector and scale levels are connected via
dense connections. For training, multiple samples should be included in the input, in which case each sample is a column of a matrix.
=#

########## External Dependencies ##########
using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, CUDA


########## Internal Dependencies ##########
#include("./Utils.jl")
#using .Utils

########## Exports ##########
export PCDense, DenseModule, DenseInitializer

########## Data structures ##########

# Dense layer. This is not callable as in Flux layers, but is used as a building block for DenseModules.
mutable struct PCDense

    states #NamedTuple{:errors, :predictions} giving the values that the layer above predicts for this layer and the prediction error
    ps #Tuple giving the learnable parameters used to predict the layer below
    grads #Tuple giving gradients of parameters to use in-place operations for weight updates
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    tc #time constant
    α #learning rate
    name
end

# Dense module. This is a callable structure that stores and updates densely connected layers in the ODE solver.
mutable struct DenseModule

    is_supervised
    inputstates
    labels
    ps
    grads
    ps2
    receptiveFieldNorms
    tc
    α
    u0
    predictions
    errors
    initializer!

end

# Callable struct that initializes u0 before running the ODE solver with a feedforward densely connected network with ReLU activations.
mutable struct DenseInitializer

    errors
    ps
    grads
    α

end

########## Constructors ##########

# Construct a dense hidden layer.
function PCDense(in_dims, out_dims, name::Symbol, T = Float32; tc = 0.1f0, α = 0.01f0)

    states = zeros(T, out_dims...)
    ps = rand(T, in_dims[1], out_dims[1])
    broadcast!(relu, ps, ps)

    grads = deepcopy(ps)
    ps2 = deepcopy(ps) .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1)
    ps ./= receptiveFieldNorms

    PCDense(states, ps, grads, ps2, receptiveFieldNorms, tc, α, name)
end



# Construct a dense module from at least three dense layers. 
function DenseModule(inputlayer, hiddenlayers, toplayer; is_supervised = false)
    
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

    initializer! = DenseInitializer(deepcopy(errors), deepcopy(ps), deepcopy(grads), α)
    
    DenseModule(is_supervised, inputstates, labels, ps, grads, ps2, receptiveFieldNorms, tc, α, u0, predictions, errors, initializer!)
end



########## Functions ##########

# Makes the dense module callable to compute the activation updates within the ODE solver. 
function (m::DenseModule)(du, u, p, t)
    
    broadcast!(relu, u, u)
    values(NamedTuple(u))[1] .= m.inputstates
    values(NamedTuple(du))[1] .= zero(eltype(du))
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(u))[end] 

    if m.is_supervised
        values(NamedTuple(u))[end] .= m.labels
    end

    m.errors .= u .- m.predictions

    for k in eachindex(m.tc)
        mul!(values(NamedTuple(m.predictions))[k], m.ps[k], values(NamedTuple(u))[k + 1])
        mul!(values(NamedTuple(du))[k + 1], transpose(m.ps[k]), values(NamedTuple(m.errors))[k])
    
        values(NamedTuple(du))[k + 1] .-= values(NamedTuple(m.errors))[k + 1]
        values(NamedTuple(du))[k + 1] .*= m.tc[k]
    end
    
    
end


# Makes the DenseInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::DenseInitializer)(u0, x)
    u0 .= 0.0f0
    values(NamedTuple(u0))[1] .= x
    for k in eachindex(m.ps)
        mul!(values(NamedTuple(u0))[k + 1], m.ps[k], values(NamedTuple(u0))[k])
        broadcast!(relu, values(NamedTuple(u0))[k + 1], values(NamedTuple(u0))[k + 1])
    end
end

# Makes PCModule callable on the integrator object of the ODE solver for training. This function updates
# each layer's parameters via a callback function.
function (m::DenseModule)(integrator)

    m.initializer!(m.u0, m.inputstates) #must call the initializer again before each training iteration because multiple training steps are taken for each run of the ODE solver
    m.initializer!.errors .= integrator.u .- m.u0

    for k in eachindex(m.tc)
        mul!(m.grads[k], values(NamedTuple(m.errors))[k], transpose(values(NamedTuple(integrator.u))[k + 1]))
        m.ps[k] .+=  m.α[k] .* m.grads[k]
        broadcast!(relu, m.ps[k], m.ps[k])
        m.ps2[k] .= m.ps[k] .^ 2
        sum!(m.receptiveFieldNorms[k], m.ps2[k])
        m.ps[k] ./= m.receptiveFieldNorms[k]
        
        mul!(m.initializer!.grads[k], values(NamedTuple(m.initializer!.errors))[k + 1], transpose(values(NamedTuple(m.u0))[k]))
        m.initializer!.ps[k] .+=  m.α[k] .* m.initializer!.grads[k]
    end

end


end