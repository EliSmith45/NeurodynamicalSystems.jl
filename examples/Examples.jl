using CairoMakie, Pkg
cd("NeurodynamicalSystems")
Pkg.activate(".")
using Revise
using NeurodynamicalSystems

using LinearAlgebra, NNlib


n = 16
m = 16
w1 = zeros(Float32, n, m)
sigma = 4
for j in axes(w1, 2)
    for i in axes(w1, 1)
        w1[i, j] = exp(-((i * m / n) - j)^2 / sigma)
    end
    normalize!(@view(w1[:, j]))
end

w1
heatmap(w1)


f = lines(w1[:, 1])
foreach(x -> lines!(w1[:, x]), axes(w1, 2))
f



y1 = zeros(Float32, n)
y1[4] = 1.0
y1[12] = 1.5
#foreach(x -> normalize!(x, 2), eachrow(w1))
G = w1' * w1
x1 =  w1' * y1
f = scatterlines(y1)
scatterlines!(x1)
f


states_base = Lca((n, m); W = w1, λ = .02, τ = .1)
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




states_rms = LcaRms((n, m); W = w1, λ = .02, τ = .1, β = .9, ϵ = .3)
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



states_adam = LcAdam((n, m); W = w1, λ = .02, τ = .01, β1 = .65, β2 = .75)
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






