module PCModules

# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./PCLayers.jl")

@reexport using .PCLayers


export PCModule, normalize_receptive_fields!, to_cpu!, to_gpu!




########## PCModules ##########

"""
    PCModule(inputlayer, hiddenlayers, u0, predictions, errors, initerror)

Predictive coding module that holds the input layer, hidden layers, initial states, predictions, errors, and initialization errors. 
The module is callable and passed as the ODE function, calculating du each iteration of the forward pass by iterating through the layers 
and calling each layer on the previous layer's states. The module is also callable on the integrator object of the ODE solver for training.

The user should never need to call a PC module directly. Just define the module and pass it to a PCNet.
"""
mutable struct PCModule

    nObs
    inputlayer
    layers
    #initializers

    predictions
    errors
    u
    du

    u0
    initerror
    ps
    psgrads
    receptiveFieldNorms
    
end



"""
    PCModule(inputlayer, hiddenlayers)

Constructor for predictive coding modules 
"""
function PCModule(inputlayer, hiddenlayers)

    nObs = size(inputlayer.data, ndims(inputlayer.data))
    layers = (inputlayer, hiddenlayers...)
    names = map(l -> l.name, layers)

    predictions, errors, u, du, u0, initerror = allocate_states(layers)
    ps, psgrads, receptiveFieldNorms = allocate_params(hiddenlayers)

    m = PCModule(nObs, inputlayer, hiddenlayers, predictions, errors, u, du, u0, initerror, ps, psgrads, receptiveFieldNorms)
    normalize_receptive_fields!(m)
    return m

end

function PCLayers.change_nObs!(m::PCModule, nObs)
    m.nObs = nObs
    map(l -> change_nObs!(l, nObs), (m.inputlayer, m.layers...))
    m.predictions, m.errors, m.u, m.du, m.u0, m.initerror = allocate_states((m.inputlayer, m.layers...))
    #m.inputlayer.data = zeros(eltype(m.inputlayer.input), size(m.inputlayer.input)[1:end-1]..., nObs)

    if values(NamedTuple(m.ps.params))[1] isa CuArray
        to_gpu!(m)
    end
end

function PCLayers.allocate_states(layers)

    names = map(l -> l.name, layers)
    predictions = NamedTuple{names}(map(l -> allocate_states(l), layers))
    predictions = ComponentArray(predictions)

    errors = deepcopy(predictions)
    u0 = deepcopy(predictions)
    initerror = deepcopy(predictions)
    u = deepcopy(predictions)
    du = deepcopy(predictions)
    
    return predictions, errors, u, du, u0, initerror
end

function PCLayers.allocate_params(hiddenlayers::Tuple)

    names = map(l -> l.name, hiddenlayers)
    params = NamedTuple{names}(map(l -> PCLayers.allocate_params(l), hiddenlayers))
    initps = NamedTuple{names}(map(l -> allocate_initparams(l), hiddenlayers))
    ps = ComponentArray(params = ComponentArray(params), initps = ComponentArray(initps))
    psgrads = deepcopy(ps)

    receptiveFieldNorms = NamedTuple{names}(map(l -> allocate_receptive_field_norms(l), hiddenlayers))
    receptiveFieldNorms = ComponentArray(receptiveFieldNorms)

    return ps, psgrads, receptiveFieldNorms
end

function PCLayers.get_gradient_activations!(m::PCModule, x)
    
    m.inputlayer.data = x
    get_gradient_activations!(values(NamedTuple(m.du))[1], values(NamedTuple(m.u))[1], m.inputlayer, values(NamedTuple(m.errors))[1])

    for k in eachindex(m.layers)
        make_predictions!(values(NamedTuple(m.predictions))[k], m.layers[k], values(NamedTuple(m.ps.params))[k], values(NamedTuple(m.u))[k + 1])
    end
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(m.u))[end]
    m.errors .= m.u .- m.predictions

    for k in eachindex(m.layers)
        get_gradient_activations!(values(NamedTuple(m.du))[k + 1], m.layers[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.errors))[k + 1], values(NamedTuple(m.ps.params))[k])
    end

end


function PCLayers.get_gradient_activations!(m::PCModule, x, y)
    
    # set input layer to x
    m.inputlayer.data = x
    get_gradient_activations!(values(NamedTuple(m.du))[1], values(NamedTuple(m.u))[1], m.inputlayer, values(NamedTuple(m.errors))[1])

    # make predictions for each layer
    for k in eachindex(m.layers)
        make_predictions!(values(NamedTuple(m.predictions))[k], m.layers[k], values(NamedTuple(m.ps.params))[k], values(NamedTuple(m.u))[k + 1])
    end

    # set the last layer's predicted value to its state so its error will be zero (as nothing predicts the top layer)
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(m.u))[end]
    m.errors .= m.u .- m.predictions

    # get the gradient of the errors with respect to the activations for each layer except the last
    for k in 1:(length(m.layers) - 1)
        get_gradient_activations!(values(NamedTuple(m.du))[k + 1], m.layers[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.errors))[k + 1], values(NamedTuple(m.ps.params))[k])
    end

    # set the last layer's state to the labels y and set its gradient to zero (as this layer is pinned to the label values)
    m.u[m.layers[end].name] .= y
    m.du[m.layers[end].name] .= 0

end

function PCLayers.get_gradient_parameters!(m::PCModule)
    for k in eachindex(m.layers)
        get_gradient_parameters!(values(NamedTuple(m.psgrads.params))[k], m.layers[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.u))[k + 1])
        get_gradient_init_parameters!(values(NamedTuple(m.psgrads.initps))[k], m.layers[k], values(NamedTuple(m.initerror))[k + 1], values(NamedTuple(m.u))[k])
    end
end

function normalize_receptive_fields!(m::PCModule)
    m.psgrads.params .= m.ps.params .^ 2
    for k in eachindex(m.layers)
        sum!(values(NamedTuple(m.receptiveFieldNorms))[k], values(NamedTuple(m.psgrads.params))[k])
        values(NamedTuple(m.ps.params))[k] ./= sqrt.(values(NamedTuple(m.receptiveFieldNorms))[k] .+ eps())
    end
end


function PCLayers.get_u0!(m::PCModule, x)
    m.inputlayer.data = x
    values(NamedTuple(m.u0))[1] .= x

    for k in eachindex(m.layers)
        get_u0!(values(NamedTuple(m.u0))[k + 1], m.layers[k], values(NamedTuple(m.ps.initps))[k], values(NamedTuple(m.u0))[k])
    end

end




function to_gpu!(m::PCModule)
    
    m.inputlayer.data = cu(m.inputlayer.data)
    m.predictions = cu(m.predictions)
    m.errors = cu(m.errors)
    m.u = cu(m.u)
    m.du = cu(m.du)
    m.u0 = cu(m.u0)
    m.initerror = cu(m.initerror)
    m.ps = cu(m.ps)
    m.psgrads = cu(m.psgrads)
    m.receptiveFieldNorms = cu(m.receptiveFieldNorms)

end

function to_cpu!(m::PCModule)
    
    m.inputlayer.data = Array(m.inputlayer.data)
    m.predictions = to_cpu!(m.predictions)
    m.errors = to_cpu!(m.errors)
    m.u = to_cpu!(m.u)
    m.du = to_cpu!(m.du)
    m.u0 = to_cpu!(m.u0)
    m.initerror = to_cpu!(m.initerror)
    m.ps = ComponentArray(params = to_cpu!(m.ps.params), initps = to_cpu!(m.ps.initps))
    m.psgrads = deepcopy(m.ps)
    
    m.receptiveFieldNorms = to_cpu!(m.receptiveFieldNorms)

end



function to_cpu!(x::ComponentArray)

    names = keys(x)
    x = ComponentArray(NamedTuple{names}(Array.(values(NamedTuple(x)))))

end

function to_cpu!(x::Array)

    x = Array(x)

end



end
