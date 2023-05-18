using CairoMakie, Pkg
cd("NeurodynamicalSystems")
Pkg.activate(".")
using Revise
using NeurodynamicalSystems

using LinearAlgebra, NNlib, Flux







n = 64
m = 64
w1 = zeros(Float32, n, m)
sigma = .025
for j in axes(w1, 2)
    for i in axes(w1, 1)
        w1[i, j] = exp(-(2 * (i - j) / n)^2 / sigma)
    end
    normalize!(@view(w1[:, j]))
end

w1
heatmap(w1)


f = lines(w1[:, 1])
foreach(x -> lines!(w1[:, x]), axes(w1, 2))
f



y1 = zeros(Float32, n)
y1[14] = 1.0
y1[44] = 1.5

x1 =  w1' * y1
f = scatterlines(y1);
scatterlines!(x1);
f


states_base = LCA((n, m); W = w1, λ = .02, τ = .1)
iters = 1000
a = zeros(eltype(y1), m, iters, 5);
u = zeros(eltype(y1), m, iters, 5);
mode = 1
for i in 1:iters
    states_base(x1)
    a[:, i, mode] .= max.(states_base.λ, states_base.u)
    u[:, i, mode] .= states_base.u
   
end



range = 1:iters
f = Figure();
ax = Axis(f[1, 1]) #scatterlines(a[1, range, mode])
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f




states_rms = LCARMS((n, m); W = w1, λ = .02, τ = .1, β = .9, ϵ = .3)
mode = 2
for i in 1:iters
    states_rms(x1)
    a[:, i, mode] .= max.(states_rms.λ, states_rms.u)
    u[:, i, mode] .= states_rms.u
end


f = Figure();
ax = Axis(f[1, 1]) #scatterlines(a[1, range, mode])
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f



states_adam = LCAAdam((n, m); W = w1, λ = .02, τ = .01, β1 = .65, β2 = .75)
mode = 3
for i in 1:iters
    states_adam(x1)
    a[:, i, mode] .= max.(states_adam.λ, states_adam.u)
    u[:, i, mode] .= states_adam.u
end


f = Figure();
ax = Axis(f[1, 1]) #scatterlines(a[1, range, mode])
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f


scatterlines(a[:, end, 1])
scatterlines(a[:, end, 2])
scatterlines(a[:, end, 3])
scatterlines(y1)





x = [1.0f0, .1f0, .68f0, .7f0]
d = size(x, 1)
β = 1
α = 1
c1 = 2*sqrt(β)
c2 = (1/(1-α)) * (β + (α^2)/2)

states_wta = WTA(d; τ = .5, α = α, β1 = sqrt(β), β2 = sqrt(β), Te = 0.0, Ti = 0.0f0, G = 1.0f0)
iters = 100
wtaOut = zeros(d + 1, iters)
for i in 1:iters
    states_wta(x)
    wtaOut[1:d, i] .= states_wta.excitatory
    wtaOut[end, i] = states_wta.inhibitory
end

sum(x)
sum(wtaOut[1:d, end])

scatterlines(wtaOut[1:end, end])
wtaOut[1:end, end]


f = Figure();
ax = Axis(f[1, 1]) 
for i in 1:(d+1)
    scatterlines!(wtaOut[i, :])
end
f



x = x1#[.1, .20f0, .25f0, .6f0, .7f0, .5, .4, .3, .25, .2, .18, .1, .05]
d = size(x, 1)
halfwidth = 3
β = 1 / (2*halfwidth + 1)
α = 1.0f0

wtaconv = WTAConv(d, halfwidth; τ = .2, α = α, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)



iters = 1000
wtaOut = zeros(d, iters, 2)
for i in 1:iters
    wtaconv(x)
    wtaOut[:, i, 1] .= wtaconv.excitatory
    wtaOut[:, i, 2] .= wtaconv.inhibitory
end

sum(x)
sum(wtaOut[:, end, 1])
sum(wtaOut[:, end, 2])

f=scatterlines(x);
scatterlines!(wtaOut[:, end, 1]);
f

f = Figure();
ax = Axis(f[1, 1]);
for i in 1:d
    scatterlines!(wtaOut[i, :, 1]);
end
f


neurons = [3, 3, 4]
a=[zeros(Bool, length(neurons)) for i in eachindex(neurons)]
        
a[2][1] = true     
a


(I * 0) * x

adam = LCAAdam((n, m); W = w1, λ = .02, τ = .01, β1 = .65, β2 = .75)
wta =  WTAConv(d, halfwidth; τ = .2, α = α, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)
crossLayerMap = [[-.2 * I, .2 * I], [.2 * I, 0 * I]]

I1 = x .+ crossLayerMap[1][1] * max.(adam.λ, adam.u) .+ crossLayerMap[1][2] * wta.excitatory 

mode = 4
for i in 1:iters
    adam(x1 .+ wta.excitatory)
    a[:, i, mode] .= max.(states_adam.λ, states_adam.u)
    u[:, i, mode] .= states_adam.u
end









mutable struct lca{T} 
    W::Matrix{T} #weights for neuron receptive fields
    G::Matrix{T} #lateral inhibition weights, G = -WWᵀ - I
    λ::T
    optimizer


    u::Array{T}
    du::Array{T}
   
end

function lca(neurons, T = Float32; W = rand(T, neurons[2], neurons[1]), λ = .01, optimizer = sgd(neurons, T = T; τ = .01))
    
    foreach(x -> normalize!(x, 2), eachcol(W))
    G = W' * W
    G[diagind(G)] .= 0
    u = zeros(T, neurons[2])
    du = copy(u)
    lca{T}(W, G, λ, optimizer, u, du)
end

# x is feedforward input signal such that W * x is feedforward stimulation
function (m::lca)(x)
    m.du = m.W' * x .- (m.G * max.(0, m.u)) .- m.u .- m.λ
    m.u = m.optimizer(m.du, m.u)
end




mutable struct sgd{T} 
    τ::T
end

function sgd(neurons, T = Float32; τ)
    sgd{T}(τ)
end

function (m::sgd)(du, u)
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



x = x1
opt1 = sgd(n, Float32; τ = .01)
lcaSGD = lca((n, m); W = w1, λ = 0.02, optimizer = opt1)

opt2 = Rmsprop(n; τ = .01, β = .9, ϵ = .3)
lcaRMS = lca((n, m); W = w1, λ = 0.02, optimizer = opt2)

opt3 = Adam(n; τ = .01, β1 = .65, β2 = .75, ϵ = eps())
lcaAdam = lca((n, m); W = w1, λ = 0.02, optimizer = opt3)

lcaa = Lca((n, m); W = w1, λ = 0.02, optimizer = "Adam", τ = .02, β1 = .65, β2 = .75, ϵ = eps())

iters = 1000
a = zeros(eltype(y1), m, iters, 3);
mode = 1
for i in 1:iters
    lcaSGD(x1)
    lcaRMS(x1)
    lcaa(x1)
    a[:, i, 1] .= max.(lcaSGD.λ, lcaSGD.u)
    a[:, i, 2] .= max.(lcaRMS.λ, lcaRMS.u)
    a[:, i, 3] .= max.(lcaa.λ, lcaa.u)
   
end




scatterlines(a[:, end, 1])
scatterlines(a[:, end, 2])
scatterlines(a[:, end, 3])

scatterlines(y1)



range = 1:iters
f = Figure();
ax = Axis(f[1, 1]) #scatterlines(a[1, range, mode])
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f








