module CompositeModules

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
export CompositeModule

########## Data structures ##########



# Dense module. This is a callable structure that stores and updates densely connected layers in the ODE solver.
mutable struct CompositeModule

    inputlayer
    layers
    u0
    predictions
    errors
    initerror

end

########## Constructors ##########


function CompositeModule(inputlayer, hiddenlayers)

    layers = (inputlayer, hiddenlayers...)

    names = map(l -> l.name, layers)
   
    errors = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
    predictions = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
    u0 = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
    initerror = deepcopy(errors)

   
    errors = ComponentArray(errors)
    predictions = ComponentArray(predictions)
    u0 = ComponentArray(u0)
    initerror = ComponentArray(initerror)
      
    CompositeModule(inputlayer, hiddenlayers, u0, predictions, errors, initerror)
end



########## Functions ##########

function (m::CompositeModule)(du, u, p, t)
    
    values(NamedTuple(u))[1] .= m.inputlayer.states
    values(NamedTuple(du))[1] .= zero(eltype(du))
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(u))[end] 

    m.errors .= u .- m.predictions

    for k in eachindex(m.layers)
        m.layers[k](values(NamedTuple(du))[k + 1], values(NamedTuple(u))[k + 1], values(NamedTuple(m.predictions))[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.errors))[k + 1])

        #mul!(values(NamedTuple(m.predictions))[k], m.ps[k], values(NamedTuple(u))[k + 1])
        #values(NamedTuple(du))[k + 1] .= mul!(values(NamedTuple(du))[k + 1], transpose(m.ps[k]), values(NamedTuple(m.errors))[k]) .- values(NamedTuple(m.errors))[k + 1]
    
    end
    
end


# Makes PCModule callable on the integrator object of the ODE solver for training. This function updates
# each layer's parameters via a callback function.
function (m::CompositeModule)(integrator)

    m.initerror .= integrator.u .- m.u0

    for k in eachindex(m.layers)
        m.layers[k](values(NamedTuple(m.errors))[k], values(NamedTuple(integrator.u))[k + 1], values(NamedTuple(m.initerror))[k + 1], values(NamedTuple(m.u0))[k])
        m.layers[k].initializer!(values(NamedTuple(m.initerror))[k + 1], values(NamedTuple(m.u0))[k], values(NamedTuple(integrator.u))[k + 1])
    end

end



    
    


end