"""
Implements all fixed point solver methods for the forward pass of PCNetworks. 
"""
module FixedPointSolvers

# External Dependencies
using LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA

# Internal Dependencies
include("./PCModules.jl")
using .PCModules

























# Exported data structures and functions
export ODEIntegrator


mutable struct ODEIntegrator
    
    odeproblem
    integrator
    u
    du
    c1
    c2

    errorLogs
    duLogs


    #kwgs
    tspan
    solver
    abstol
    reltol
    save_everystep
    save_start
    dt
    adaptive
    dtmax
    dtmin

end

function ODEIntegrator(pcmodule; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false, dt = 0.1f0, adaptive = true, dtmax = 1.0f0, dtmin = 0.01f0)
    
    ode = ODEProblem(pcmodule, pcmodule.u0, tspan, Float32[])
    integrator = init(ode, solver, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = save_start, dt = dt, adaptive = adaptive, dtmax = dtmax, dtmin = dtmin, alias_u0 = false)
    u = integrator.u
    du = get_du(integrator)
    c1 = deepcopy(u)
    c2 = deepcopy(u)

    errorLogs = eltype(pcmodule.u0)[]
    duLogs = eltype(pcmodule.u0)[]
    ODEIntegrator(ode, integrator, u, du, c1, c2, errorLogs, duLogs, tspan, solver, abstol, reltol, save_everystep, save_start, dt, adaptive, dtmax, dtmin)

end

function (m::ODEIntegrator)(maxSteps::Int, stoppingCondition)
    
    m.errorLogs = zeros(eltype(m.u), maxSteps)
    m.duLogs = zeros(eltype(m.u), maxSteps)

    #run the ODE solver while logging intermediate steps
    for i in 1:maxSteps
        step!(m.integrator)
        m.c1 .= abs.(m.odeproblem.f.f.errors)
        m.errorLogs[i] = sum(m.c1)
        
        m.c1 .= abs.(m.du)
        m.c2 .= m.u .^ 2

        m.duLogs[i] = dot(m.c1, m.u) / (eps() + sum(m.c2))

        if m.duLogs[i] < stoppingCondition 
            m.errorLogs = @view(m.errorLogs[1:i])
            m.duLogs = @view(m.duLogs[1:i])
            break
        end

    end


    return m.u

end


function (fps::ODEIntegrator)(x::Bool)

    reinit!(fps.integrator, reinit_cache = true)

end





end