module NeurodynamicalSystems


########## External Dependencies ##########
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA

########## Internal Dependencies ##########
include("./PCModules.jl")
include("./PCNetworks.jl")
#include("./GPUUtils.jl")
include("./Utils.jl")

using .PCModules
using .PCNetworks
#using .GPUUtils
import .Utils: gaussian_basis, sample_basis

########## Exports ##########
export PCDense, PCConv, PCInput, CompositeModule, PCNet
export train!, reset!, to_gpu!, to_cpu!
export gaussian_basis, sample_basis





########## Moving modules, initializers, and networks to/from gpu ##########

function to_gpu!(x::PCInput)

    x.states = cu(x.states)

end

function to_gpu!(x::PCDense)

   
    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
    x.ps2 = cu(x.ps2)
    x.receptiveFieldNorms = cu(x.receptiveFieldNorms)
    x.tc = cu(x.tc)
    x.α = cu(x.α)
    
    x.labels = cu(x.labels)
    to_gpu!(x.initializer!)

end


function to_gpu!(x::PCConv)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
    x.ps2 = cu(x.ps2)
    x.receptiveFieldNorms = cu(x.receptiveFieldNorms)
    x.tc = cu(x.tc)
    x.α = cu(x.α)
    
    x.labels = cu(x.labels)
    to_gpu!(x.initializer!)

end

function to_gpu!(x::CompositeModule)

    to_gpu!(x.inputlayer)

    for hl in x.layers
        to_gpu!(hl)
    end

  
    x.u0 = cu(x.u0)
    x.predictions = cu(x.predictions)
    x.errors = cu(x.errors)
    x.initerror = cu(x.initerror)
  
end

function to_gpu!(x::DenseInitializer)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
    #x.α = cu(x.α)
    #x.errors = cu(x.errors)
   
end



function to_gpu!(x::ConvInitializer)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
   # x.α = cu(x.α)
   
end


function to_gpu!(x::PCNet)
    to_gpu!(x.odemodule)
    x.odeprob = ODEProblem(x.odemodule, x.odemodule.u0, (0.0f0, 1.0f0), Float32[])
    x.sol = solve(x.odeprob, BS3(), abstol = 0.01f0, reltol = 0.01f0, save_everystep = false, save_start = false)

end


function to_cpu!(x::ComponentArray)

    names = keys(x)
    x = ComponentArray(NamedTuple{names}(Array.(values(NamedTuple(x)))))

end



function to_cpu!(x::DenseInitializer)

    x.ps = Array.(x.ps)
    x.grads = Array.(x.grads)
    x.α = Array(x.α)
    x.errors = to_cpu!(x.errors)
   
end


function to_cpu!(x::ConvInitializer)

    x.ps = Array.(x.ps)
    x.grads = Array.(x.grads)
    x.α = Array(x.α)
    x.errors = to_cpu!(x.errors)

end


function to_cpu!(x::PCNet)
    to_cpu!(x.odemodule)
    x.odeprob = ODEProblem(x.odemodule, x.odemodule.u0, (0.0f0, 1.0f0), Float32[])
    x.sol = solve(x.odeprob, BS3(), abstol = 0.01f0, reltol = 0.01f0, save_everystep = false, save_start = false)

end



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

