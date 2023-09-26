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

end

########## Functions ##########
function PCNet(m)

    m.is_supervised = false
  
    ode = ODEProblem(m, m.u0, (0.0f0, 1.0f0), Float32[])
    sol = solve(ode, BS3(), abstol = 0.01f0, reltol = 0.01f0, save_everystep = false, save_start = false)

    PCNet(m, ode, sol)
end


# Makes PCNetwork callable for unsupervised learning. This sets the input parameters to x before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), abstol = 1e-4, reltol = 1e-2, save_everystep = false)
    
    m.odemodule.is_supervised = false
    m.odemodule.inputstates .= x
    m.odemodule.initializer!(m.odeprob.u0, x)

    m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = false)
end


# Makes PCNetwork callable for supervised learning. This sets the input parameters to x and pins the top level to the targets y before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}, y::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), abstol = 1e-4, reltol = 1e-2, save_everystep = false)
    
    m.odemodule.is_supervised = true
    m.odemodule.inputstates .= x
    m.odemodule.labels .= y
    m.odemodule.initializer!(m.odeprob.u0, x)

    m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = false).u[1]
end


# Set all states to zero to run the network on a new input
function reset!(pcn::PCNet)

    pcn.odemodule.u0 .= zero(eltype(pcn.odemodule.errors))
    pcn.odeprob.u0 .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.predictions .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.errors .= zero(eltype(pcn.odemodule.errors))
    pcn.odemodule.initializer!.errors .= zero(eltype(pcn.odemodule.errors))
    
end

# Trains the PCNetwork in an unsupervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), iters = 1, stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], abstol = abstol, reltol = reltol, save_everystep = false)
    
    m.odemodule.is_supervised = false
    m.odemodule.inputstates .= x
    
    cb = DiscreteCallback((u, t, integrator) -> t in stops, integrator -> m.odemodule(integrator))

    for i in 1:iters

        reset!(m)
        m.odemodule.initializer!(m.odeprob.u0, x)
        m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, callback = cb, save_everystep = save_everystep, save_start = false)
        
        if i % 10 == 0
            error = sum((m.odemodule.errors) .^ 2)
            println("Iteration $i squared error: $error")
        end
        
    end
end

# Trains the PCNetwork in a supervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::Union{Array, CuArray}, y::Union{Array, CuArray}, tspan = (0.0f0, 100.0f0); solver = BS3(), iters = 1, stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], abstol = abstol, reltol = reltol, save_everystep = false)
    
    m.odemodule.is_supervised = true
    m.odemodule.inputstates .= x
    m.odemodule.labels .= y

    cb = DiscreteCallback((u, t, integrator) -> t in stops, integrator -> m.odemodule(integrator))

    for i in 1:iters
        reset!(m)
        m.odemodule.initializer!(m.odeprob.u0, x)
        m.sol = solve(m.odeprob, solver, tspan = tspan, abstol = abstol, reltol = reltol, callback = cb, save_everystep = save_everystep, save_start = false)
        
        if i % 10 == 0
            error = sum((m.odemodule.errors) .^ 2)
            println("Iteration $i squared error: $error")
        end
        
    end
end


end