module PCNetworks

#=
Callable Predictive Coding Network objects. These store any PC module, and can be called on an input to run the corresponding ODE system.
=#

########## Internal Dependencies ##########
include("./PCModules.jl")
using .PCModules

########## External Dependencies ##########
using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, CUDA

########## Exports ##########
export PCNet, reset!, train!
#export PCInput #from InputLayers.jl
#export PCDense, DenseModule, DenseInitializer #from DenseModules.jl
#export PCConv, ConvModule, ConvInitializer #from ConvModules.jl


########## Data structures ###########

# PCNetwork contains a PCModule and ODE solver arguments. Calling it on an input runs the full ODE solver for the chosen time span.
mutable struct PCNet

    odemodule
    odeprob
    sol
    iter
    initerror
    error

end

########## Functions ##########
function PCNet(m)

   # m.is_supervised = false
  
    ode = ODEProblem(m, m.u0, (0.0f0, 1.0f0), Float32[])
    sol = solve(ode, BS3(), abstol = 0.01f0, reltol = 0.01f0, save_everystep = false, save_start = false)

    PCNet(m, ode, sol, [1], [0.0f0], [0.0f0])
end


# Set all states to zero to run the network on a new input
function reset!(pcn::PCNet)

    pcn.odemodule.u0 .= zero(eltype(pcn.odemodule.errors))
    pcn.odeprob.u0 .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.predictions .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.errors .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.initerror .= zero(eltype(pcn.odemodule.errors))
    
end

# Makes PCNetwork callable for unsupervised learning. This sets the input parameters to x before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}, tspan = (0.0f0, 10.0f0); solver = BS3(), abstol = 0.01f0, reltol = 0.05f0, save_everystep = false)
    
    m.odemodule.inputlayer.states .= x
    values(NamedTuple(m.odemodule.u0))[1] .= x

    for k in eachindex(m.odemodule.layers)
        m.odemodule.layers[k].is_supervised = false
        m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.u0))[k + 1], values(NamedTuple(m.odemodule.u0))[k])
    end

    m.odeprob = ODEProblem(m.odemodule, m.odemodule.u0, tspan, Float32[])
    m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = false)
end


# Makes PCNetwork callable for supervised learning. This sets the input parameters to x and pins the top level to the targets y before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}, y::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), abstol = 0.01f0, reltol = 0.05f0, save_everystep = false)
   
    m.odemodule.inputlayer.states .= x
    values(NamedTuple(m.odeprob.u0))[1] .= x

    for k in 1:(length(m.odemodule.layers) - 1)
        m.odemodule.layers[k].is_supervised = false
        m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.u0))[k + 1], values(NamedTuple(m.odemodule.u0))[k])
    end

    m.odemodule.layers[end].is_supervised = true
    m.odemodule.layers[end].initializer!(values(NamedTuple(m.odemodule.u0))[end], values(NamedTuple(m.odemodule.u0))[end - 1])
    m.odemodule.layers[end].labels = y

    m.odeprob = ODEProblem(m.odemodule, m.odemodule.u0, tspan, Float32[])
    m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = false)
end



# Trains the PCNetwork in an unsupervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), iters = 1, stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], decayLrEvery = 500, lrDecayRate = 0.85f0, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.05f0, save_everystep = false)
    
    m.iter = collect(1:iters)
    m.initerror = zeros(eltype(x), length(m.iter))
    m.error = zeros(eltype(x), length(m.iter))

    m.odemodule.inputlayer.states .= x
    cb = DiscreteCallback((u, t, integrator) -> t in stops, integrator -> m.odemodule(integrator))

    for i in 1:iters

        #initialize the network
        values(NamedTuple(m.odemodule.u0))[1] .= x
        for k in eachindex(m.odemodule.layers)
            m.odemodule.layers[k].is_supervised = false
            m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.u0))[k + 1], values(NamedTuple(m.odemodule.u0))[k])
        end

        #run the ODE solver with a callback to train the PC layers
        m.odeprob = ODEProblem(m.odemodule, m.odemodule.u0, tspan, Float32[])
        m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, callback = cb, save_everystep = save_everystep, save_start = false)
        

        #calculate the initializer error and update its parameters
        m.odemodule.initerror .= m.sol.u[end] .- m.odemodule.u0
        for k in eachindex(m.odemodule.layers)
            m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.initerror))[k + 1], values(NamedTuple(m.odemodule.u0))[k], values(NamedTuple(m.sol.u[end]))[k + 1])
        end


        m.error[i] = sum(values(NamedTuple((m.odemodule.errors)))[1] .^ 2)
        m.initerror[i] = sum(values(NamedTuple((m.odemodule.initerror)))[end] .^ 2)
          
        if i % decayLrEvery == 0
            for k in eachindex(m.odemodule.layers)
                m.odemodule.layers[k].α *= lrDecayRate
                m.odemodule.layers[k].initializer!.α *= lrDecayRate
            end
        end

        #print results
        if i % show_every == 0
            println("Iteration $i squared error: $(m.error[i])")
        end
        
        if i % normalize_every == 0
            for k in eachindex(m.odemodule.layers)
                m.odemodule.layers[k].ps2 .= m.odemodule.layers[k].ps .^ 2
                sum!(m.odemodule.layers[k].receptiveFieldNorms, m.odemodule.layers[k].ps2)
                m.odemodule.layers[k].ps ./= m.odemodule.layers[k].receptiveFieldNorms
            end
        end
        reset!(m)
    end
end

# Trains the PCNetwork in a supervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::Union{Array, CuArray}, y::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), iters = 1, stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], decayLrEvery = 500, lrDecayRate = 0.85f0, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.05f0, save_everystep = false)
    
    m.iter = collect(1:iters)
    m.initerror = zeros(eltype(x), length(m.iter))
    m.error = zeros(eltype(x), length(m.iter))

   
    m.odemodule.inputlayer.states .= x
    cb = DiscreteCallback((u, t, integrator) -> t in stops, integrator -> m.odemodule(integrator))
    m.odemodule.layers[end].labels = y
  

    for i in 1:iters

        #initialize the network
        values(NamedTuple(m.odemodule.u0))[1] .= x
        for k in 1:(length(m.odemodule.layers) - 1)
            m.odemodule.layers[k].is_supervised = false
            m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.u0))[k + 1], values(NamedTuple(m.odemodule.u0))[k])
        end
    
        m.odemodule.layers[end].is_supervised = true
        m.odemodule.layers[end].initializer!(values(NamedTuple(m.odemodule.u0))[end], values(NamedTuple(m.odemodule.u0))[end - 1])
        m.odemodule.layers[end].labels .= y
    
        #run the ODE solver with a callback to train the PC layers
        m.odeprob = ODEProblem(m.odemodule, m.odemodule.u0, tspan, Float32[])
        m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, callback = cb, save_everystep = save_everystep, save_start = false)



        #calculate the initializer error and update its parameters
        m.odemodule.initerror .= m.sol.u[end] .- m.odemodule.u0
        for k in eachindex(m.odemodule.layers)
            m.odemodule.layers[k].initializer!(values(NamedTuple(m.odemodule.initerror))[k + 1], values(NamedTuple(m.odemodule.u0))[k], values(NamedTuple(m.sol.u[end]))[k + 1])
        end

        m.error[i] = sum(values(NamedTuple((m.odemodule.errors)))[1] .^ 2)
        m.initerror[i] = sum(values(NamedTuple((m.odemodule.initerror)))[end] .^ 2)

        if i % decayLrEvery == 0
            for k in eachindex(m.odemodule.layers)
                m.odemodule.layers[k].α *= lrDecayRate
                m.odemodule.layers[k].initializer!.α *= lrDecayRate
            end
        end

        #print results
        if i % show_every == 0
            #error = sum(values(NamedTuple((m.odemodule.errors)))[1] .^ 2)
            println("Iteration $i squared error: $(m.error[i])")
        end
        
        if i % normalize_every == 0
            for k in eachindex(m.odemodule.layers)
                m.odemodule.layers[k].ps2 .= values(NamedTuple(m.odemodule.layers[k].ps)) .^ 2
                sum!(m.odemodule.layers[k].receptiveFieldNorms, m.odemodule.layers[k].ps2)
                m.odemodule.layers[k].ps ./= m.odemodule.layers[k].receptiveFieldNorms
            end
        end

        reset!(m)
    end
end


end