module NeurodynamicalSystems


using LinearAlgebra


#Locally competitive algorithm for highly efficient biologically-plausible sparse coding:
#
# Kaitlin L. Fair, Daniel R. Mendat, Andreas G. Andreou, Christopher J. Rozell,
# Justin Romberg and David V. Anderson. "Sparse Coding Using the Locally Competitive 
# Algorithm on the TrueNorth Neurosynaptic System"
# https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full



# X: data matrix with rows as samples
# G: correlation/inhibition strength of neurons. For an IIR filterbank, g(i, j) this is the 
# frequency response of filter i at center frequency wj
# τ: time constant, should be about 10ms
# iters: how many times neurons are updated for each time sample. Should be low (as low as 1) if 
# the audio sample rate is high (e.g. fs > 20kHz)
# returns sparse code A, representing a highly sparse time frequency distribution with implicit total 
# variation minimization. 

export Lca



mutable struct Lca{T} 
    W::Matrix{T} #weights for neuron receptive fields
    G::Matrix{T} #lateral inhibition weights, G = -WWᵀ - I
    λ::T
    optimizer


    u::Array{T}
    du::Array{T}
   
end

function Lca(neurons, T = Float32;
    W = rand(T, neurons[2], neurons[1]), 
    λ = .01, 
    optimizer = "Adam", 
    τ = .01, 
    β1 = .65, 
    β2 = .75, 
    ϵ = eps())
    
    foreach(x -> normalize!(x, 2), eachcol(W))
    G = W' * W
    G[diagind(G)] .= 0
    u = zeros(T, neurons[2])
    du = copy(u)


    if optimizer == "SGD"
        opt = Sgd(T; τ)
    elseif optimizer == "RMS"
        opt = Rmsprop(neurons[2], T; τ, β = β1, ϵ)
    elseif optimizer == "Adam"
        opt = Adam(neurons[2], T; τ, β1, β2, ϵ)
    else
        println("Invalid optimizer, choosing Adam with default parameters")
        opt = Adam(neurons[2], T; τ = .1, β1 = .65, β2 = .75, ϵ = eps())
    end


    Lca{T}(W, G, λ, opt, u, du)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::Lca)(x)
    m.du = m.W' * x .- (m.G * max.(m.λ, m.u)) .- m.u .- m.λ
    m.u = m.optimizer(m.du, m.u)
end




mutable struct Sgd{T} 
    τ::T
end

function Sgd(T = Float32; τ)
    Sgd{T}(τ)
end

function (m::Sgd)(du, u)
    u .+ (m.τ .* du)
end

mutable struct Rmsprop{T} 
    τ::T
    β::T
    ϵ::T
    v::Array{T}
end

function Rmsprop(neurons, T = Float32; τ, β, ϵ)
    v = zeros(T, neurons)
    Rmsprop{T}(τ, β, ϵ, v)
end

function (m::Rmsprop)(du, u)
    m.v = (m.v .* m.β) .+ (1 - m.β) .* (du .^ 2)
    u .+ ((m.τ ./ (m.v .+ m.ϵ)) .* du)
end


mutable struct Adam{T} 
    τ::T
    β1::T
    β2::T
    ϵ::T
    m::Array{T}
    v::Array{T}
end


function Adam(neurons, T = Float32; τ, β1, β2, ϵ)
    m = zeros(T, neurons)
    v = copy(m)
    Adam{T}(τ, β1, β2, ϵ, m, v)
end

function (m::Adam)(du, u)
    m.m = (m.β1 .* m.m) .+ (1 - m.β1) .* (du)
    m.v = (m.β2 .* m.v) .+ (1 - m.β2) .* (du .^ 2)
    u .+ ((m.τ / m.β1) .* (m.m ./ (sqrt.(m.v ./ m.β2) .+ eps())))
end


#Locally competitive algorithm as a stateful function similar to Flux.jl RNNs
mutable struct WTA{T} 


    τ::T
    α::T
    β1::T
    β2::T
    Te::T
    Ti::T
    G::T

    excitatory::Array{T}
    inhibitory::T

   
   
end

export WTA
function WTA(neurons, T = Float32;  τ = .01f0, 
                                    α = .5f0,
                                    β1 = 1.0f0,
                                    β2 = 1.0f0,
                                    Te = .010f0,
                                    Ti = .010f0,
                                    G = 1.0f0)
    
   
    excitatory = zeros(T, neurons)
    inhibitory = zero(T)

    WTA{T}(τ, α, β1, β2, Te, Ti, G, excitatory, inhibitory)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::WTA)(x)
    m.inhibitory = m.inhibitory .+ (m.τ * (max.(0, (m.β2 * sum(m.excitatory)) - m.Ti) - (m.G * m.inhibitory)))
    m.excitatory = m.excitatory .+ (m.τ .* (max.(0, x .+ (m.α  .* m.excitatory) .- ((m.β1 * m.inhibitory ) + m.Te)) .- (m.G .* m.excitatory)))
    
end


export LocallyConnectedWTA
#locally competitive algorithm as a stateful function similar to Flux.jl RNNs
mutable struct LocallyConnectedWTA{T} 

    W::Matrix{T}
    τ::T
    α::T
    β::T
    Te::T
    Ti::T
    G::T

    excitatory::Array{T}
    inhibitory::Array{T}

   
   
end


function LocallyConnectedWTA(neurons, halfwidth, T = Float32;  τ = .1f0, 
                                    α = .5f0,
                                    β = 1.0f0,
                                  
                                    Te = .010f0,
                                    Ti = .010f0,
                                    G = 1.0f0)
    
   
    excitatory = zeros(T, neurons)
    inhibitory = copy(excitatory)
    W = zeros(T, neurons, neurons)

    for j in axes(W, 2)
        for i in axes(W, 1)
            if abs(i - j) <= halfwidth
                W[i, j] = sqrt(β)
            end
        end
    end
    LocallyConnectedWTA{T}(W, τ, α, β, Te, Ti, G, excitatory, inhibitory)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::LocallyConnectedWTA)(x)
    m.inhibitory = m.inhibitory .+ (m.τ * (max.(0, (m.W * m.excitatory) .- m.Ti) .- (m.G * m.inhibitory)))
    m.excitatory = m.excitatory .+ (m.τ .* (max.(0, x .+ (m.α  .* m.excitatory) .- (m.W * m.inhibitory ) .- m.Te) .- (m.G .* m.excitatory)))
  
end



export LcaWTA
#locally competitive algorithm as a stateful function similar to Flux.jl RNNs
mutable struct LcaWTA{T} 

    lcaLayer::Lca{T}
    wtaLayer::LocallyConnectedWTA{T}
    flowRates::Vector{Vector{T}}
    inputs::Vector{Vector{T}}
end


function LcaWTA(lca, wta, T = Float32; flowRates = [[-.10f0, .10f0], [.10f0, -.10f0]])
    
   
    inputs = [zeros(T, length(lca.u)), zeros(T, length(wta.excitatory))]
    
    LcaWTA{T}(lca, wta, flowRates, inputs)
end


# x is feedforward input signal such that W * x is feedforward stimulation
function (m::LcaWTA)(x)

    m.inputs[1] = x .+ (m.flowRates[1][1] .* max.(m.lcaLayer.λ, m.lcaLayer.u)) .+ (m.flowRates[1][2] .* m.wtaLayer.excitatory)
    m.inputs[2] = (m.flowRates[2][1] .* m.lcaLayer.u) .+ (m.flowRates[2][2] .* m.wtaLayer.excitatory)
   
    m.lcaLayer(m.inputs[1])
    m.wtaLayer(m.inputs[2])
    
end





mutable struct Dnn1{T} #block-sparse matrix
  
    connected::Vector{Vector{Bool}}
    transforms::Vector{Vector{Any}}
    scales::Vector{Vector{T}}
    thresholds::Vector{Vector{T}}
    tcs::Vector{Vector{T}} #time constants

    layers::Vector{Vector{T}}
    updates::Vector{Vector{T}}
end

function Dnn1(neurons, T = Float32; 
    tcs = zeros(T, length(neurons)), 
    thresholds = zeros(T, length(neurons)),
    connected = nothing, 
    transforms = nothing,
    scales = nothing)
    
    layers = [zeros(T, neurons[i]) for i in eachindex(neurons)]
    updates = deepcopy(layers)
    if isnothing(connected)
        connected = [zeros(Bool, length(neurons)) for i in eachindex(neurons)]
        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if (j - i) == 1
                    connected[j][i] = true
                end
                
            end
        end
        
    end

    if isnothing(transforms)
        transforms = [[[0 0 ; 0 0] for j in eachindex(neurons)] for i in eachindex(neurons)]

        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if connected[j][i] 
                    transforms[j][i] = rand(T, neurons[j], neurons[i])
                end
                
            end
        end
        
    end

    if isnothing(scales)
        scales = [zeros(length(neurons)) for i in eachindex(neurons)]
        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if connected[j][i] 
                    scales[j][i] = rand(T)
                end
                
            end
        end
        
    end

    Dnn1{T}(connected, transforms, scales, thresholds, tcs, layers, updates)
 
end







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











function (m::Dnn1)()
   
    @views for j ∈ eachindex(m.W)
        m.U[j] .+= m.τ[j] .* ((m.W[j]' * m.A).- m.U[j] )
      
    end

    
    @views for j ∈ 2:length(m.A)
        hard_threshold!(m.A[j], m.U[j], m.λ[j]) 
    end
end

function (m::Dnn1)(x)
    m.A[1] .= x
end


end
