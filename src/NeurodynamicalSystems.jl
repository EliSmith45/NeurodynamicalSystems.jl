module NeurodynamicalSystems


using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, Random

export nonneg_normalized!, gaussian_basis, sample_basis


########## Data types for proper dispatch of update functions for predictive coding layers

# PCLayer{A} is the type of layer in a predictive coding network. {A} must be TopLayer, HiddenLayer, or InputLayer depending on its position in the hierarchy.
# TopLayers predict the layer below but are at the top of the hierarchy and therefore are not predicted by any layers. With each time step, their activations are 
# changed only to better predict the layers below.
# HiddenLayers predict the layer below and are predicted by the layer above. At each time step, their activations are changed to simultaneously better predict the 
# layer below, and to be more consistent with the predictions of the layer above. This allows for top-down error-correction of sparse codes similar to hopfield networks
# and transformers.
# InputLayers contain the input data and have no learnable parameters. They can either be fixed to the input data, or updated to be more consistent with the data and 
# predictions from the layers above. This allows for hopfield-like error correction when the data is noisy or partially obscured.

abstract type PCLayer{A} end 
abstract type TopLayer end 
abstract type HiddenLayer end
abstract type InputLayer end




########## Linear predictive coding layers ##########

mutable struct PCLinearLayer{A} <: PCLayer{A}

    states #NamedTuple{:errors, :predictions} giving the values that the layer above predicts for this layer and the prediction error
    inputstates #NamedTuple{:errors, :predictions} giving the values that this layer predicts for the layer below and the prediction error
    ps #NamedTuple giving the learnable parameters used to predict the layer below
    tc #time constant
    α #learning rate
    name::Symbol #layer name wrapped in Val() for indexing the component arrays containing the activations

end

# Construct a linear hidden layer.
function PCLinearLayer(in_dims, out_dims, name::Symbol, T = Float32; tc = 0.1f0, α = 0.01f0)

    states = (predictions = zeros(T, out_dims...), errors = zeros(T, out_dims...))
    inputstates = (predictions = zeros(T, in_dims...), errors = zeros(T, in_dims...))
    
    weights = rand(T, in_dims[1], out_dims[1])
    nonneg_normalized!(weights)

    grads = copy(weights)
    ps = (weight = weights, grads = grads)

    PCLinearLayer{HiddenLayer}(states, inputstates, ps, tc, α, name, Val(name))
end

# Constructs a linear top layer 
function PCLinearTop(in_dims, out_dims, name::Symbol, T = Float32; tc = 0.1f0, α = 0.01f0)

    
    states = (predictions = zeros(T, out_dims...), errors = zero(T))
    inputstates = (predictions = zeros(T, in_dims...), errors = zeros(T, in_dims...))
    
    weights = rand(T, in_dims[1], out_dims[1])
    nonneg_normalized!(weights)

    grads = copy(weights)
    ps = (weight = weights, grads = grads)

    PCLinearLayer{TopLayer}(states, inputstates, ps, tc, α, name, Val(name))
end

# Constructs a linear input layer
function PCLinearInput(out_dims, name::Symbol, T = Float32; tc = 0.1f0, α = nothing)

    states = (predictions = zeros(T, out_dims...), errors = zeros(T, out_dims...))
    inputstates = nothing
    ps = (input = copy(states.predictions),)

    PCLinearLayer{InputLayer}(states, inputstates, ps, tc, α, name, Val(name))
end




########## Convolutional predictive coding layers ##########

mutable struct PCConvLayer{A} <: PCLayer{A}

    states
    inputstates
    ps
    cdims
    tc
    α
    name
    valname

end

# Constructs a convolutional hidden layer
function PCConvLayer(k, ch_in, ch_out, in_dims, name::Symbol, T = Float32; stride=1, padding=0, dilation=1, groups=1, tc = 0.1f0, α = 0.01f0)

    inputstates = (predictions = zeros(T, in_dims...), errors = zeros(T, in_dims...))
    weights = rand(T, k..., ch_in, ch_out)
    nonneg_normalized!(weights)

    grads = copy(weights)
    ps = (weight = weights, grads = grads)

    cdims = NNlib.DenseConvDims(size(inputstates.errors), size(weights); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    y = NNlib.conv(inputstates.errors, weights, cdims)
    
    states = (predictions = copy(y), errors = copy(y))
    
    PCConvLayer{HiddenLayer}(states, inputstates, ps, cdims, tc, α, name, Val(name))
end

# Constructs a convolutional top layer
function PCConvTopLayer(k, ch_in, ch_out, in_dims, name::Symbol, T = Float32; stride=1, padding=0, dilation=1, groups=1, tc = 0.1f0, α = 0.01f0)

    inputstates = (predictions = zeros(T, in_dims...), errors = zeros(T, in_dims...))
    weights = rand(T, k..., ch_in, ch_out)
    nonneg_normalized!(weights)

    grads = copy(weights)
    ps = (weight = weights, grads = grads)
    cdims = NNlib.DenseConvDims(size(inputstates.errors), size(weights); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    y = NNlib.conv(inputstates.errors, weights, cdims)
   
    states = (predictions = copy(y), errors = zero(T))
    
    PCConvLayer{TopLayer}(states, inputstates, ps, cdims, tc, α, name, Val, name)
end

# Constructs a convolutional input layer
function PCConvInputLayer(in_dims, name::Symbol, T = Float32; tc = 0.1f0, α = nothing)

    states = (predictions = zeros(T, in_dims...), errors = zeros(T, in_dims...))
    inputstates = nothing
    ps = (input = copy(states.predictions),)
    cdims = nothing
    
    PCConvLayer{InputLayer}(states, inputstates, ps, cdims, tc, α, name, Val(name))
end


# Makes PCLayers callable for the update steps of the ODE solver. Takes the layer's du and u elements,
# then updates du and other layer states in-place. This function should never be called on its own but 
# is called behind the scenes by the ODE solver

function (m::PCLayer)(du, u)
    #u .= relu(u)
    update_errors!(m, u)
    update_du!(du, m, u)
    update_predictions!(m, u)
end



########## Functions to update states and parameters for each layer type ##########
# As before, none of these should be called by the user. 

function update_errors!(m::PCLayer{W}, u) where W <: Union{HiddenLayer, InputLayer}
    m.states.errors .= u .- m.states.predictions
end

function update_errors!(m::PCLayer{TopLayer}, u) 
    return
end

function update_du!(du, m::PCLinearLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    du .= m.tc .* ((m.ps.weight' * m.inputstates.errors) .- m.states.errors)
end
function update_du!(du, m::PCConvLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    NNlib.conv!(du, m.inputstates.errors, m.ps.weight, m.cdims)
    du .-= m.states.errors
    du .*= m.tc
end
function update_du!(du, m::PCLayer{InputLayer}, u)
    du .= m.tc .* (m.ps.input .- u)
end

function update_predictions!(m::PCLinearLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    mul!(m.inputstates.predictions, m.ps.weight, u)
end
function update_predictions!(m::PCConvLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    NNlib.∇conv_data!(m.inputstates.predictions, u, m.ps.weight, m.cdims)
end
function update_predictions!(m::PCLayer{InputLayer}, u) 
    return
end

function update_weights!(m::PCLinearLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    m.ps.weight .+= mul!(m.ps.grads, m.inputstates.errors, (m.α .* u)')
    nonneg_normalized!(m.ps.weight)
end

function update_weights!(m::PCConvLayer{W}, u) where W <: Union{HiddenLayer, TopLayer}
    NNlib.∇conv_filter!(m.ps.grads, m.inputstates.errors, m.α .* u, m.cdims)
    #m.ps.weight .+= m.α .* m.ps.grads
    nonneg_normalized!(m.ps.weight)
end

function update_weights!(m::PCLayer{InputLayer}, u) 
    return
end





########## PCModule functions ##########

# PCModules contain a named tuple of layers and are callable. Calling them iterates through the layers and calls them
# on each layer's component of the du and u ComponentArrays to compute the overall update step. 
mutable struct PCModule3
    
    layers

end

# Links a tuple of layers by setting inputstates of each layer to states of the layer below. 
# This memory is shared between layers rather than copied, so it makes indexing much easier without
# expanding the overall memory requirements of the network.
function link_layers!(layers)
    
    u0 = map(l -> l.states.predictions, layers)
    names = map(l -> l.name, layers)

    for i in 2:length(names)
        layers[i].inputstates = layers[i - 1].states
    end
    
    layers = NamedTuple{names}(layers)
    u0 = ComponentArray(NamedTuple{names}(u0))
    PCModule3(layers), u0, ()

end

# Makes the PCModule callable to compute the activation updates
function (m::PCModule3)(du, u, p, t)
    
    u .= relu(u)
    foreach(layer -> layer(@view(du[layer.valname]), @view(u[layer.valname])), m.layers)
    
end

# Makes PCModule callable on the integrator object of the ODE solver. This function updates
# each layer's parameters via a callback function.
function (m::PCModule3)(integrator)
    
    foreach(layer -> update_weights!(layer, @view(u[layer.valname])), m.layers)
   
end





########## PCNetwork functions ##########

# PCNetwork contains a PCModule and ODE solver arguments. Calling it on an input runs the full ODE solver for the 
# chosen time span.
mutable struct PCNet1
    pcmodule
    u0
    ps
    solver
    saveat
    reltol
    abstol
    callback
end

function PCNet1(pcmodule, u0, ps; solver = Tsit5(), saveat = 1.0f0, reltol = 1e-6, abstol = 1e-6, callback = 0.01f0)
    PCNet1(pcmodule, u0, ps, solver, saveat, reltol, abstol, callback)
end

# Makes PCNetwork callable. This sets the input parameters to x before running the ODE system.
function (m::PCNet1)(x, tspan = (0.0f0, 1.0f0); saveat = tspan[2])
    m.pcmodule.layers.L0.ps.input .= x
    ode = ODEProblem(m.pcmodule, m.u0, tspan, m.ps)
    solve(ode, m.solver, saveat = saveat).u[end]
end

# Set all states to zero to run the network on a new input
function reset!(pcn::PCNet1)

    for k in keys(pcn.pcmodule.layers)
        pcn.pcmodule.layers[k].states.predictions .= 0
        pcn.pcmodule.layers[k].states.errors .= 0
    end
    
end

# Trains the PCNetwork. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train(net::PCNet1, x, tspan = (0.0f0, 1.0f0); stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], saveat = tspan[2])
    
    net.pcmodule.layers.L0.ps.input .= x
    ode = ODEProblem(net.pcmodule, net.u0, tspan, net.ps)
        
    cb = DiscreteCallback((u, t, integrator) -> t in stops, integrator -> net.pcmodule(integrator))


    solve(ode, Tsit5(), callback = cb, tstops = stops, saveat = saveat).u[end]

end



########## Utility functions ##########


# Force all elements of an array to be nonnegative and normalize features by their L2 norm. 
# These functions are called during training to normalize weights and enforce nonnegativity, 
# which helps with sparse coding. 

function nonneg_normalized!(weight::AbstractArray{T, 2}) where T
    
    for k in axes(weight, 2)
        weight[:, k] .= relu.(weight[:, k])
        weight[:, k] ./= norm(weight[:, k], 2)
    end
    
end

function nonneg_normalized!(weight::AbstractArray{T, 3}) where T
        
    for k in axes(weight, 3)
        weight[:, :, k] .= relu.(weight[:, :, k])
        weight[:, :, k] ./= norm(weight[:, :, k], 2)
    end

end

function nonneg_normalized!(weight::AbstractArray{T, 4}) where T
        
    for k in axes(weight, 4)
        weight[:, :, :, k] .= relu.(weight[:, :, :, k])
        weight[:, :, :, k] ./= norm(weight[:, :, :, k], 2)
    end

end

function nonneg_normalized!(weight::AbstractArray{T, 5}) where T
        
    for k in axes(weight, 5)
        weight[:, :, :, :, k] .= relu.(weight[:, :, :, :, k])
        weight[:, :, :, :, k] ./= norm(weight[:, :, :, :, k], 2)
    end

end


# Creates a gaussian basis to represent dummy features for toy problems
function gaussian_basis(n, m; basesCenters = (1/n):(1/n):1, binCenters = (1/m):(1/m):1, sigma = 1.0, T = Float32)
    
    w = zeros(T, m, n)
  
    for (j, basis) in enumerate(basesCenters)
        for (i, bin) in enumerate(binCenters)
            w[i, j] = exp(-((basis - bin) / (2 * sigma)) ^ 2); #* (exp(-((centers[end] - centers[j]) / (2 * sigma)) ^ 2) - exp(-((centers[end] - centers[j]) / (2 * sigma)) ^ 2))
        end
    end

    foreach(x -> normalize!(x, 2), eachcol(w))

    w
end

# Generate a dummy data set from a linear combination of features. This is for toy problems to 
# evaluate the sparse codes and learned features.
function sample_basis(basis; nObs = 1, nActive = 2, maxCoherence = .999)

    G = basis' * basis
    n = size(basis, 2)
    m = size(basis, 1)

    if nObs == 1

        y = zeros(eltype(basis), n)
        x = zeros(eltype(basis), m)

    else
        y = zeros(eltype(basis), n, nObs)
        x = zeros(eltype(basis), m, nObs)
    end
    a = 1
  
    
    for t in nObs
        possible = collect(axes(basis, 2))
        #println(possible)
        for i in 1:nActive

            j = rand(possible)
            y[j] = rand(.5:.001:1)
            possibleNew = []
            for k in possible
                if G[j, k] < maxCoherence
                    append!(possibleNew, k)
                end
            end

            possible = possibleNew


        end

    end

    return basis * y, y


end


#=

#Cappa and LiquidRnn are experimental!

#"CAPPA: Continuous-time Accelerated Proximal Point Algorithm for Sparse Recovery" https://arxiv.org/pdf/2006.02537.pdf
#https://link.springer.com/article/10.1007/s00521-022-08166-5

mutable struct Cappa{T} 
    W::Union{Matrix{T}, UniformScaling{Bool}} #neuron feedforward receptive fields
    G::Matrix{T} #weights for lateral inhibition, Wl = -WxWxᵀ - I

    λ::T
    η::T
    α1::T
    α2::T
    k1::T
    k2::T
    


    f::Array{T}
    step::Array{T}
    z::Array{T}
    znorm::T
    dx::Array{T}
    x::Array{T}
   
end

function Cappa(neurons, T = Float32; W = rand(T, neurons, neurons),  λ = .01, η = .5, α1 = .5, α2 = 2, k1 = .5, k2 = .5)
    G = W * W'
    f = zeros(T, neurons)
    step = copy(f)
    z = copy(f)
    znorm = 0
    dx = copy(f)
    x = copy(f)
    Cappa{T}(W, G, λ, η, α1, α2, k1, k2, f, step, z, znorm, dx, x)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::Cappa)(y)


    m.f = m.G * m.x .- m.W' * y
    m.step = m.x .- m.η .* m.f
    m.z = m.x .- hard_threshold.(m.step, m.η * m.λ)
    m.znorm = norm(m.z)
    
    m.dx .= (m.k1 .* (m.z) ./ ((m.znorm) ^ (1 - m.α1))) .+ (m.k2 .* (m.z) ./ ((m.znorm) ^ (1 - m.α2)))
    m.x .+= m.dx

  
end











mutable struct LiquidRnn12{T} #block-sparse matrix
    W0::Matrix{T} #input weights
    W1::Matrix{T} #recurrent weights
    bw::Array{T} #weight matrix bias
    b::Array{T} #recurrent layer biases
    τ::T #recurrent layer biases

    U::Array{T}
    A::Array{T}

    delta::T
end

function LiquidRnn12(neurons, n_observations, T = Float32; W0 = rand(T, neurons[2], neurons[1]), W1 = rand(T, neurons[2], neurons[2]), bw = rand(T, neurons[2], n_observations), b = rand(T, neurons[2], n_observations), τ = .010f0, delta = .01)
    U = zeros(T, neurons[2], n_observations)
    A = copy(U)
    LiquidRnn12{T}(W0, W1, bw, b, τ, U, A, delta)
end


function (m::LiquidRnn12)(x)
    m.A .=  tanh.((m.W1 * m.U) .+ (m.W0 * x) .+ m.b) 
    m.U .= (m.U .+ (m.delta .* m.A .* m.bw) ) ./ (1 .+ (m.delta .* ((1 / m.τ) .+ m.A)))
    hard_threshold!(m.U, copy(m.U), 0.0f0)
end








=#
end
