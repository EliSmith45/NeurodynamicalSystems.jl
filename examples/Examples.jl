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

lcaSGD = Lca((n, m); W = w1, λ = 0.02, optimizer = "SGD", τ = .1)
lcaRMS = Lca((n, m); W = w1, λ = 0.02, optimizer = "RMS", τ = .1, β1 = .65, ϵ = .3)
lcaAdam = Lca((n, m); W = w1, λ = 0.02, optimizer = "Adam", τ = .01, β1 = .55, β2 = .7, ϵ = eps())

halfwidth = 3
β = 1 / (2*halfwidth + 1)
α = 1.0f0

wtaLocal = LocallyConnectedWTA(d, halfwidth; τ = .2, α = 1.0f0, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)
lc = Lca((n, m); W = w1, λ = 0.02, optimizer = "Adam", τ = .01, β1 = .55, β2 = .7, ϵ = eps())
lcawta = LcaWTA(lc, wtaLocal; flowRates = [[-.10f0, .5], [.5, 0.0f0]])

iters = 150
a = zeros(eltype(y1), m, iters, 6);

for i in 1:iters
    lcaSGD(x1)
    lcaRMS(x1)
    lcaAdam(x1)
    lcawta(x1)
    a[:, i, 1] .= max.(lcaSGD.λ, lcaSGD.u)
    a[:, i, 2] .= max.(lcaRMS.λ, lcaRMS.u)
    a[:, i, 3] .= max.(lcaAdam.λ, lcaAdam.u)
    a[:, i, 4] .= max.(lcawta.lcaLayer.λ, lcawta.lcaLayer.u)
    a[:, i, 5] .= lcawta.wtaLayer.excitatory
    a[:, i, 6] .= lcawta.wtaLayer.inhibitory
   
end




scatterlines(a[:, end, 1])
scatterlines(a[:, end, 2])
scatterlines(a[:, end, 3])
scatterlines(a[:, end, 4])
scatterlines(a[:, end, 5])
scatterlines(a[:, end, 6])


scatterlines(y1)



range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]) #scatterlines(a[1, range, mode])
mode = 1;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f

range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]); #scatterlines(a[1, range, mode])
mode = 2;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f


range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]); #scatterlines(a[1, range, mode])
mode = 3;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f




range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]); #scatterlines(a[1, range, mode])
mode = 4;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f


range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]); #scatterlines(a[1, range, mode])
mode = 6;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f






x = [1.0f0, .1f0, .68f0, .7f0]
d = size(x, 1)
β = 1
α = 1
c1 = 2*sqrt(β)
c2 = (1/(1-α)) * (β + (α^2)/2)

wta = WTA(d; τ = .5, α = α, β1 = sqrt(β), β2 = sqrt(β), Te = 0.0, Ti = 0.0f0, G = 1.0f0)
iters = 100
wtaOut = zeros(d + 1, iters)
for i in 1:iters
    wta(x)
    wtaOut[1:d, i] .= wta.excitatory
    wtaOut[end, i] = wta.inhibitory
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
halfwidth = 5
β = 1 / (2*halfwidth + 1)
α = 1.0f0

wtaconv = LocallyConnectedWTA(d, halfwidth; τ = .2, α = α, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)



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




