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

export Lca, LcaRms, LcAdam, Cappa

#Locally competitive algorithm as a stateful function similar to Flux.jl RNNs
mutable struct Lca{T} 
    W::Matrix{T} #weights for neuron receptive fields
    G::Matrix{T} #lateral inhibition weights, G = -WWᵀ - I
    λ::T
    τ::T


    u::Array{T}
   
end

function Lca(neurons, T = Float32; W = rand(T, neurons[2], neurons[1]), λ = .01, τ = .01)
    
    foreach(x -> normalize!(x, 2), eachcol(W))
    G = W' * W
    G[diagind(G)] .= 0
    u = zeros(T, neurons[2])

    Lca{T}(W, G, λ, τ, u)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::Lca)(x)
    m.u .+= m.τ .* (m.W' * x .- (m.G * max.(m.λ, m.u)) .- m.u)
end




#locally competitive algorithm where the (T)ransform W is fixed/known and
#learning rate is chosen adaptively as in RMSProp/ADAM
mutable struct LcaRms{T} 
    W::Matrix{T} #feedforward stimulation
    G::Matrix{T} #lateral inhibition weights, G = -WWᵀ - I
    λ::T
    τ::T
    β::T
    ϵ::T


    u::Array{T}
    du::Array{T}
    v::Array{T}
end

function LcaRms(neurons, T = Float32; W = rand(T, neurons[2], neurons[1]), λ = .01, τ = .01, β = .9, ϵ = .001)    
    
    foreach(x -> normalize!(x, 2), eachcol(W))
    G = W' * W
    G[diagind(G)] .= 0
    u = zeros(T, neurons[2])
    du = copy(u)
    v = copy(u) #.+ 1.0f0
    LcaRms{T}(W, G, λ, τ, β, ϵ, u, du, v)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::LcaRms)(x)
    m.du .= m.W' * x .- (m.G * max.(m.λ, m.u)) .- m.u 
    m.v .*= m.β
    m.v .+= (1 - m.β) .* (m.du .^ 2)

    m.u .+= (m.τ ./ (m.v .+ m.ϵ)) .* m.du

end


#locally competitive algorithm where the (T)ransform W is fixed/known and
#learning rate is chosen adaptively as in RMSProp/ADAM
mutable struct LcAdam{T} 
    W::Matrix{T} #feedforward stimulation
    G::Matrix{T} #lateral inhibition weights, G = -WWᵀ - I
    λ::T
    τ::T
    β1::T
    β2::T
    ϵ::T


    u::Array{T}
    du::Array{T}
    m::Array{T}
    v::Array{T}
end

function LcAdam(neurons, T = Float32; W = rand(T, neurons[2], neurons[1]), λ = .01, τ = .01, β1 = .9, β2 = .9)    
    
    foreach(x -> normalize!(x, 2), eachcol(W))
    G = W' * W
    G[diagind(G)] .= 0
    u = zeros(T, neurons[2])
    du = copy(u)
    m = copy(u)
    v = copy(u)
    LcAdam{T}(W, G, λ, τ, β1, β2, eps(), u, du, m, v)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::LcAdam)(x)
    m.du .= m.W' * x .- (m.G * max.(m.λ, m.u)) .- m.u 
    m.m .*= m.β1
    m.m .+= (1 - m.β1) .* (m.du)


    m.v .*= m.β2
    m.v .+= (1 - m.β2) .* (m.du .^ 2)


    m.u .+= (m.τ / m.β1) .* (m.m ./ (sqrt.(m.v ./ m.β2) .+ eps()))
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







#Locally competitive algorithm as a stateful function similar to Flux.jl RNNs
mutable struct LcaSelfAttention{T} 
    Wx::Union{Matrix{T}, UniformScaling{Bool}} #neuron feedforward receptive fields
    Wl::Matrix{T} #weights for lateral inhibition, Wl = -WxWxᵀ - I
    Wg::Matrix{T} #Weights for neurons' unique information, Wg = (Wl / ||Wl||) + I
    Wa::Matrix{T} #Weights for self-attention-like computation, Wa = WxWxᵀ


    λ::T
    τ::T
    β::T
    ϵ::T
    α::T


    A::Matrix{T}
    u::Array{T}
    a::Array{T}
    du::Array{T}
    v::Array{T}
    g::Array{T}
end

function LcaSelfAttention(neurons, T = Float32; Wx = I, Wl = rand(T, neurons, neurons), Wg = rand(T, neurons, neurons), Wa = rand(T, neurons, neurons), λ = .01, τ = .01, β = .9, ϵ = .001, α  = .1)
    
    A = copy(Wa)
    u = zeros(T, neurons)
    a = copy(u)
    du = copy(u)
    v = copy(u)
    g = copy(u)
    LcaSelfAttention{T}(Wx, Wl, Wg, Wa, λ, τ, β, ϵ, α, A, u, a, du, v, g)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::LcaSelfAttention)(x)

    for j in axes(m.A, 2)
        m.A[:, j] .= exp.(@view(m.Wa[:, j]) .* m.a)
        m.A[:, j] ./= sum(@view(m.A[:, j]))
    end
    m.g = m.α .* tanh.((1/m.α) * m.Wg * m.a) .* m.a 
    m.du .= x - (m.Wl * m.a) .- m.u .+ m.g
    m.v .*= m.β
    m.v .+= (1 - m.β) .* m.du .^2

    m.u .+= (m.τ ./ (m.v .+ m.ϵ)) .* (m.A' * m.du)
    
    hard_threshold!(m.a, m.u, m.λ)

  
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








mutable struct Dnn{T} #block-sparse matrix
    W::Vector{Vector}
    linked::Vector{Vector{Bool}}

   
    #dU::Array{Array{T}}
    U::Array{Array{T}}
    A::Array{Array{T}}

    λ::Array{T}
    τ::Array{T}


end

function Dnn(neurons, n_observations, T = Float32; W = nothing, linked = nothing, λ = repeat([.01f0], inner = length(neurons)), τ = repeat([.01f0], inner = length(neurons)))
    
  
    if isnothing(linked)
        linked = [zeros(Bool, length(neurons)) for i in eachindex(neurons)]
        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if abs(i - j) <= 1
                    linked[j][i] = true
                end
                
            end
        end
        
    end

    

    

    if isnothing(W)
        W = initW(neurons, linked, T)
        
    end
  
   
   
   # dU = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    U = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    A = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]

  
    Dnn{T}(W, linked,  U, A, λ, τ)
 
end








function hard_threshold!(x::Array{T}, vals::Array{T}, threshold::Number) where T <: Real
   
    for i in eachindex(vals)
        if vals[i] < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end

function hard_threshold!(x::Array{T}, threshold::Number) where T <: Real
   
    for i in eachindex(x)
        if x[i] < threshold
            x[i] = 0
        end
    end
end


function hard_threshold(vals, threshold::Number) where T <: Real
   

    if vals < threshold
        0
    else
        vals
    end
    
end

export convn, fastconv

##############################################
# Generic convn function using direct method for computing convolutions:
# Accelerated Convolutions for Efficient Multi-Scale Time to Contact Computation in Julia
# Alexander Amini, Alan Edelman, Berthold Horn
##############################################
using Base.Cartesian
@generated function convn(E::Array{T,N}, k::Array{T,N}) where {T,N}
    quote
        sizeThreshold = 21;
        if length(k) <= sizeThreshold || $N > 2
            #println("using direct")
            retsize = [size(E)...] + [size(k)...] .- 1
            retsize = tuple(retsize...)
            ret = zeros(T, retsize)

            convn!(ret,E,k)
            return ret
        elseif $N == 2 #greater than threshold but still compatible with base julia
            #println("using fft2")
            return conv2(E,k)
        else
            #println("using fft1")
            return conv(E,k)
        end
    end
end

# direct version (do not check if threshold is satisfied)
@generated function fastconv(E::Array{T,N}, k::Array{T,N}) where {T,N}
    quote

        retsize = [size(E)...] + [size(k)...] .- 1
        retsize = tuple(retsize...)
        ret = zeros(T, retsize)

        convn!(ret,E,k)
        return ret

    end
end


# in place helper operation to speedup memory allocations
@generated function convn!(out::Array{T}, E::Array{T,N}, k::Array{T,N}) where {T,N}
    quote
        @inbounds begin
            @nloops $N x E begin
                @nloops $N i k begin
                    (@nref $N out d->(x_d + i_d - 1)) += (@nref $N E x) * (@nref $N k i)
                end
            end
        end
        return out
    end
end



function (m::Dnn)()
   
    @views for j ∈ eachindex(m.W)
        m.U[j] .+= m.τ[j] .* ((m.W[j]' * m.A).- m.U[j] )
      
    end

    
    @views for j ∈ 2:length(m.A)
        hard_threshold!(m.A[j], m.U[j], m.λ[j]) 
    end
end

function (m::Dnn)(x)
    m.A[1] .= x
end


end
