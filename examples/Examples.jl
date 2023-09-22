using Pkg, CairoMakie
cd("NeurodynamicalSystems");
Pkg.activate(".");
using Revise;
using NeurodynamicalSystems;

using LinearAlgebra, NNlib, ComponentArrays, DifferentialEquations, CUDA, SparseArrays


# Generate synthetic data from a given basis. The bases are discretely sampled Gaussians,
# which is common in image and audio processing. These are 1D but could easily be generalized
# to higher dimensions. 

# The goal is to sparsely encode this data. Each input will contain a small number of active bases,
# and we want to determine the correct activities of each basis for a given input. This inverse 
# problem is the fundamental multivariate problem that nearly all neural networks aim to solve, 
# regardless of the type of input signal or network architecture. 

# The chosen basis is extremely coherent, meaning that they are highly correlated. Sparsely encoding 
# such signals remains an open problem, so hopefully these networks can accomplish it. We will first assume
# the bases are known, i.e., the network is already trained. Later we'll see how well these networks can 
# learn the bases

n = 5000
r = 41
a = rand(Float32, n, n)
for j in axes(a, 2)
    for i in axes(a, 1)
        if abs(i - j) >= r
            a[i, j] = 0
        end
    end
end
a
b = sparse(a)

@time b * a

c = cu(b)
ac = cu(a)

@time d = c * ac;
@time d = c' * ac;
@time transpose(c);

dd = ComponentArray(L1 = c, L2 = deepcopy(c))



n = 64; #number of bases
m = 64; 
nObs = 1

sigma = Float32(1/n); #width of each Gaussian
w = gaussian_basis(n, m; sigma = sigma) #make gaussian basis

x, y = sample_basis(w; nObs = nObs, nActive = 2, maxCoherence = .99) #sample from the basis
y


heatmap(w)
f = scatterlines(y[:, 1]);
scatterlines!(x[:, 1]);

f






n0 = m
n1 = 64
n2 = 64

l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; tc = 1.0f0, α = 0.005f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2;  tc = 1.0f0, α = 0.005f0)

mo = DenseModule(l0, (l1,), l2, is_supervised = false)
pcn = PCNet(mo)


pcn.odemodule.ps[1] .= w
pcn.odemodule.ps[2] .*= 0
pcn.odemodule.ps[2][diagind(pcn.odemodule.ps[2])] .= 1.0f0
pcn.odemodule.initializer!.ps[1] .*= 0
pcn.odemodule.initializer!.ps[2] .*= 0

@time yh = pcn(x, (0.0f0, 10.0f0), abstol = 0.005f0, reltol = 0.01f0);

obs = 1
yh
scatterlines(yh.L1[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(pcn.odemodule.predictions.L0[:, obs])
f

scatterlines(yh.L2[:, obs])
scatterlines(y[:, obs])


reset!(pcn)
@time train!(pcn, x, (0.0f0, 150.0f0); iters = 100, abstol = 0.005f0, reltol = 0.01f0, stops = 140.0f0:2.0f0:150.0f0)


ssp = SteadyStateProblem(pcn.odemodule, pcn.odemodule.u0, Float32[])
@time sssol = solve(ssp, DynamicSS(BS3(), abstol = 1e-3, reltol = 1e-2, tspan = Inf))
scatterlines(sssol.u.L1[:, 1])

@time sssol = solve(ssp, SSRootfind())
scatterlines(sssol.u.L1[:, 1])



fu = (du, u) -> pcn.odemodule(du, u, 0.0f0, Float32[])
scatterlines(pcn.odemodule.inputstates[:, 1])

@time nls = nlsolve(fu, pcn.odemodule.u0)
scatterlines(nls.zero.L1[:, 1])



n0 = m
n1 = 256
n2 = 256
n3 = 256
n4 = 256
n5 = 256
n6 = 256
n7 = 256
n8 = 256
n9 = 256

l0 = PCInput((n0, nObs), :L0)
l1 = PCDense((n0, nObs), (n1, nObs), :L1; tc = .1f0, α = 0.005f0)
l2 = PCDense((n1, nObs), (n2, nObs), :L2;  tc = .1f0, α = 0.005f0)
l3 = PCDense((n2, nObs), (n3, nObs), :L3;  tc = .12f0, α = 0.005f0)
l4 = PCDense((n3, nObs), (n4, nObs), :L4;  tc = .15f0, α = 0.005f0)
l5 = PCDense((n4, nObs), (n5, nObs), :L5;  tc = .2f0, α = 0.005f0)
l6 = PCDense((n5, nObs), (n6, nObs), :L6;  tc = .25f0, α = 0.005f0)
l7 = PCDense((n6, nObs), (n7, nObs), :L7;  tc = .25f0, α = 0.005f0)
l8 = PCDense((n7, nObs), (n8, nObs), :L8;  tc = .25f0, α = 0.005f0)
l9 = PCDense((n8, nObs), (n9, nObs), :L9;  tc = .25f0, α = 0.005f0)

mo = DenseModule(l0, (l1, l2, l3, l4), l9)
pcn = PCNet(mo)


@time yh = pcn(x, (0.0f0, 50.0f0), abstol = 0.005f0, reltol = 0.01f0);

obs = 1
yh
scatterlines(yh.L1[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(pcn.odemodule.predictions.L0[:, obs])
f

scatterlines(yh.L2[:, obs])
scatterlines(y[:, obs])


reset!(pcn)
@time train!(pcn, x, (0.0f0, 150.0f0); iters = 100, abstol = 0.005f0, reltol = 0.01f0, stops = 140.0f0:2.0f0:150.0f0)




to_gpu!(pcn)
xc = cu(x)

@time yh = pcn(xc, (0.0f0, 100.0f0), abstol = 0.005f0, reltol = 0.01f0);


yhc = Array.(values(NamedTuple(yh)))
obs = 1
scatterlines(Array(yhc[2])[:, obs])

f = scatterlines(x[:, obs])
scatterlines!(Array(pcn.odemodule.predictions.L0)[:, obs])
f

scatterlines(Array(yh.L2)[:, obs])
scatterlines(y[:, obs])








scatterlines(pcn.odemodule.errors.L1)

@time train!(pcn, xc, (0.0f0, 10.0f0); iters = 1000, abstol = 0.005f0, reltol = 0.02f0, stops = 8.0f0:0.1f0:10.0f0)


pcn.odemodule.ps.L2


N = 1000
M = 30
a = Tuple([cu(rand(Float32, N, N)) for i in 1:M])
b = Tuple([cu(rand(Float32, N, N)) for i in 1:M])

c = deepcopy(a)

A = ArrayPartition(a)
B = ArrayPartition(b)
C = ArrayPartition(c)



@time mul!.(C.x, adjoint.(A.x), B.x);

@time BLAS.gemm!('T', 'N', 1.0f0, A.x[1], B.x[1], 0.0f0, C.x[1])

C.x[1]


C[1]
vcat(collect(C.x))
C = A .+ B
Ac = cu(A)
Ac.x
A.x[1]



xc = cu(x)
m = pcn.odemodule
m.inputstates = xc
du = deepcopy(m.u0)
u = deepcopy(m.u0)

m.errors.L0

m(du, u, m.gpuindicator, 0.0f0)
scatterlines(Array(du.L1))
u .+= du

@time @cuda threads = m.nthreads blocks = m.nblocks dense_predict_update!(values(NamedTuple(du)), values(NamedTuple(u)), values(NamedTuple(m.predictions)), values(NamedTuple(m.errors)), values(NamedTuple(m.ps)), m.constants.tc)

function cutest(a, b)

    i = ((blockIdx().x - 1) * blockDim().x) + threadIdx().x

    if i <= length(a)
        q = 1
        while b[q] < i
            q += 1
        end
        a[i] = b[q]
    end

    return
end


a = (cu(zeros(3)), cu(ones(3)), 2 .* cu(ones(3)))
b = (0, 3, 6)

@cuda threads = length(a) blocks = 1 cutest(a, b)
a
b



findfirst(x -> x > 3, b)


broadcast!(relu, u, u)
@view(u[m.names[1]]) .= m.inputstates
@view(du[m.names[1]]) .*= 0

@view(m.predictions[m.names[end]]) .= @view(u[m.names[end]])

m.errors .= u .- m.predictions

m.errors.L1

u.L0

heatmap(pcn.odemodule.ps.L1)
#solve the ODE system for input data x and timespan ts, updating weights once at each time in stops.


for i in 1:100
    train(pcn, x1, (0.0f0, 100.0f0); stops = 95.0f0:1.0f0:100.0f0)
    reset!(pcn)
end    

heatmap(pcn.pcmodule.layers.L1.ps.weight')
#(u, t, integrator) -> SciMLBase.get_du!
reset!(pcn)

@time yh = pcn(x1, (0.0f0, 100.0f0))

scatterlines(x)
scatterlines(vec(yh.L1))

scatterlines(y)

pl1 = pcn.pcmodule.layers.L1.ps.weight
scatterlines(pl1[15, 15:30])

f = scatterlines(pl1[15:30, 15])

for i in axes(pl1, 2)[16:30]
    scatterlines!(pl1[15:30, i])
end
f


f = scatterlines(pl1[22:38, 22])

for i in axes(pl1, 2)[22:38]
    scatterlines!(pl1[22:38, i])
end
f


function initLCA(n, m, T = Float32; W = rand(T, n, m), tc = 0.1f0, tspan = (0.0f0, 1.0f0))
    
    
    
    u0 = zeros(T, m)
   
    input = zeros(T, n)
    ps = ComponentArray(input = input, W = W)
    p = ComponentArray(tc = tc, ps = ps)
    
    ode = ODEProblem(lca!, u0, tspan, p)
    return  u0, p, ode

end
function initLCAConv(n1, n2, k, ch, T = Float32; W = rand(T, k[1], k[2], ch[1], ch[2]), tc = 0.1f0, tspan = (0.0f0, 1.0f0))
    
    
    
    u0 = zeros(T, n1, n2, ch[1], ch[2])
    c = Conv(k, ch, identity; pad = SamePad(), use_bias = false)
    ct = ConvTranspose(k, ch, identity; pad = SamePad(), use_bias = false)


    rng = Random.default_rng()
    Random.seed!(rng, 0)
    @time weights, st = Lux.setup(rng, c)
    Lux.setup(rng, ct)
    weights.weight .= W

    input = copy(u0)
    ps = ComponentArray(input = input, W = weights, st = st)
    p = (tc = tc, c = c, ct = ct, ps = ps)
    




    ode = ODEProblem(lcaconv!, u0, tspan, p)
    return  u0, p, ode

end
function initCappa(n, m, T = Float32; W = rand(T, n, m), tc = .10f0, k1 = 0.5f0, a1 = .5f0, k2 = .50f0, a2 =1.5f0, tspan = (0.0f0, 1.0f0))
    
    
    proj = zeros(T, n)
    state = zeros(T, m)
    
    u0 = ComponentArray(proj = proj, states = state)
    input = zeros(T, n)
    ps = ComponentArray(input = input, W = W)
    p = ComponentArray(tc = tc, k1 = k1, a1 = a1, k2 = k2, a2 = a2, ps = ps)
    
    ode = ODEProblem(cappa!, u0, tspan, p)
    return  u0, p, ode

end

function initLCARouted(n, m, T = Float32; W = rand(T, n, m), tc = 1.0f0, tspan = (0.0f0, 1.0f0))
    
    
    proj = zeros(T, n)
    states = zeros(T, m)
    pred = zeros(T, m, n)
    u0 = ComponentArray(proj = proj, states = states, predictions = pred)
    input = copy(res)
    ps = ComponentArray(input = input, W = W)
    p = ComponentArray(tc = tc, ps = ps)
    
    ode = ODEProblem(lcaRouted!, u0, tspan, p)
    return  u0, p, ode

end


function lca!(du, u, p, t)
   
    @unpack tc, ps = p
    @unpack input, W = ps

    du .= tc .* W' * (input .- W * relu.(u))
   
end
function lcaconv!(du, u, p, t)
    #@unpack l1, l2 = u
    @unpack input, W = ps
   
    du .= tc .* ct(input .- c(relu.(u), W, NamedTuple())[1], W, NamedTuple())[1]
   
end

@time a = Conv((8, 1), (1 => 1), identity; pad = SamePad(), bias = false)


rng = Random.default_rng()
Random.seed!(rng, 0)
@time ps, st = Lux.setup(rng, a)


x1 = reshape(x, n, 1, 1, 1)
b = a(x1, ps, st)[1]

conv(relu.(x1), ww)


function cappa!(du, u, p, t)
    @unpack proj, states = u
    @unpack tc, k1, a1, k2, a2, ps = p
    @unpack input, W = ps

    du.proj .= tc .* (W' * (input .- W * relu.(states)))
    l2norm = norm(du.proj) 

    du.states .= k1 .* (du.proj ./ (eps() .+ l2norm ^ (1 - a1))) .+ k2 .* (du.proj ./ (eps() .+ l2norm ^ (1 - a2)))
end

function lcaRouted!(du, u, p, t)
    @unpack proj, states, predictions = u
    @unpack tc, ps = p
    @unpack input, W = ps

    du.proj .= (input .- W * relu.(states))
    
    du.predictions .=  W' .* relu.(states) .- predictions
    du.states .= tc .* ((W' .* softmax(predictions)) * proj)
end




a = [1.0 2; 3 4]
b= [1.0, 2]

a .* b


tspan = (0.0f0, 43.0f0)
saveat = 23.0f0
u0, p, ode = initLCA(n, m; W = w, tspan = tspan)
ode.p.ps.input .= copy(x)

@time yh = solve(ode, Tsit5()).u[end];
lines(yh)


u0, p, ode = initCappa(n, m; W = w, tspan = tspan, tc = .15f0, k1 = 1.5f0, a1 = .99f0, k2 = 12.0f0, a2 = 1.5f0)
ode.p.ps.input .= copy(x)

u0, p, ode = initLCARouted(n, m; W = w, tc = 1.0f0, tspan = tspan)
ode.p.ps.input .= copy(x)

krn = reshape(w[20:40, 30], 21, 1, 1, 1)
lines(krn)
aa=(1 => 1)

u0, p, ode = initLCAConv(n, 1, (21, 1), (1 => 1); W = krn, tc = 0.1f0, tspan = tspan)
ode.p.ps.input .= reshape(x1, n, 1, 1, 1)

@time yh = solve(ode, Tsit5()).u[end];
yh = reshape(yh, n)
lines(yh)



lines(input)
lines(y)
lines(yh[end].proj)








function forward(weights)

    ode.p.ps.W .= weights.W
    yhat = relu.(solve(ode, Tsit5(), saveat = saveat).u[end].states)
    z = 0
    return yhat, z
end


wg = ComponentArray(W = w)
yy = forward(wg)
lines(yy[1])

function conditions(weights, yhat, z)
    du = weights.W' * (ode.p.ps.input .- weights.W * relu.(yhat))
    return (yhat .- .+ du) 

end

implicit = ImplicitFunction(forward, conditions)

yy = (first ∘ implicit)(wg)
lines(yy)



grad = Zygote.jacobian(first ∘ implicit, wg)[1]


function predict_ode!(m, x, saveat = 1.0f0)
    m.p.sensor.input .= x;
    solve(m, Tsit5(), saveat = saveat).u[end];
end
function loss(m, x, y)
    yhat = predict_ode!(m, x)
    sum(abs, yhat.sparse.h .- y)
end

tc = .3f0
tspan = (0.0f0, 1.0f0)
ode = LCA3(n, m; w = w, tc = tc, tspan = tspan)

@time sol = predict_ode!(ode, x);

lines(x)
lines(y)
lines(sol)


lca1 = LCA3(n, m; name = :lca1);
lcaprob = ODEProblem(lca1.odeSys, u0, (0.0f0, 1.0f0));
tspan = (0.0f0, 1.0f0)

lca1.odeSys.states[1]









lcaSGD = LCA((n, m); W = w1, λ = 0.02, optimizer = "SGD", τ = .1)
lcaRMS = LCA((n, m); W = w1, λ = 0.02, optimizer = "RMS", τ = .1, β1 = .65, ϵ = .3)
lcaAdam = LCA((n, m); W = w1, λ = 0.02, optimizer = "Adam", τ = .01, β1 = .55, β2 = .7, ϵ = eps())

iters = 150
a = zeros(eltype(y1), m, iters, 6);

for i in 1:iters
    lcaSGD(x1)
    lcaRMS(x1)
    lcaAdam(x1)
    #lcawta(x1)
    a[:, i, 1] .= max.(lcaSGD.λ, lcaSGD.u)
    a[:, i, 2] .= max.(lcaRMS.λ, lcaRMS.u)
    a[:, i, 3] .= max.(lcaAdam.λ, lcaAdam.u)
    #a[:, i, 4] .= max.(lcawta.lcaLayer.λ, lcawta.lcaLayer.u)
    #a[:, i, 5] .= lcawta.wtaLayer.excitatory
    #a[:, i, 6] .= lcawta.wtaLayer.inhibitory
   
end




scatterlines(a[:, end, 1])
scatterlines(a[:, end, 2])
scatterlines(a[:, end, 3])
scatterlines(a[:, end, 4])
scatterlines(a[:, end, 5])
scatterlines(a[:, end, 6])


scatterlines(y1)



#plot regular LCA
range = 1:iters;
f = Figure();
ax = Axis(f[1, 1]) 
mode = 1;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f

#LCA-RMSProp
f = Figure();
ax = Axis(f[1, 1]); 
mode = 2;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode])
end
f
 mutable struct hm{T}

    x::T
 end

 mutable struct bfv{T} <: hu
    x::T
 end


#LCA-Adam
f = Figure();
ax = Axis(f[1, 1]); 
mode = 3;
for j in axes(a, 1)
    scatterlines!(a[j, range, mode]);
end
f




####### WTA networks
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
halfwidth = 8
β = 1 / (2*halfwidth + 1)
α = 1.0f0

wtaconv = LocallyConnectedWTA(d, halfwidth; τ = .02, α = α, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)



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
scatterlines!(wtaOut[:, end, 2]);
f


f = Figure();
ax = Axis(f[1, 1]);
for i in 1:d
    scatterlines!(wtaOut[i, :, 1]);
end
f





halfwidth = 3
β = 1 / (2*halfwidth + 1)
α = 1.0f0

wtaLocal = LocallyConnectedWTA(n, halfwidth; τ = .2, α = 1.0f0, β = β, Te = 0.0, Ti = 0.0f0, G = 1.0f0)
lc = LCA((n, m); W = w1, λ = 0.02, optimizer = "Adam", τ = .01, β1 = .55, β2 = .7, ϵ = eps())
lcawta = LcaWTA(lc, wtaLocal; flowRates = [[-.10f0, .5], [.5, 0.0f0]])

for i in 1:iters

    lcawta(x1)
    a[:, i, 4] .= max.(lcawta.lcaLayer.λ, lcawta.lcaLayer.u)
    a[:, i, 5] .= lcawta.wtaLayer.excitatory
    a[:, i, 6] .= lcawta.wtaLayer.inhibitory
   
end


scatterlines(a[:, end, 4])
scatterlines(a[:, end, 5])
scatterlines(a[:, end, 6])





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
using Flux


a = rand(15, 1, 1)
cn = Conv(rand(3, 1, 3); pad = SamePad())
b = cn(a)
cnt = ConvTranspose(rand(3, 1, 3); pad = SamePad())
cnt(b)

# higher order features
pairwise = zeros(eltype(w1), n^2, m)

for j in axes(pairwise, 2)
    pairwise[:, j] .= reshape(w1[:, j] * w1[:, j]', n^2, 1)
end

pairwiseCor = pairwise' * pairwise
heatmap(w1' * w1)
heatmap(pairwiseCor)