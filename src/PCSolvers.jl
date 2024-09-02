##### Fixed Point Solvers #####
module PCSolvers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./PCModules.jl")

@reexport using .PCModules

export EulerSolver, ForwardEulerSolver, BackwardEulerSolver
export HeunSolver, ForwardHeunSolver, BackwardHeunSolver
export forwardSolverStep!, backwardSolverStep!, forwardSolve!, change_step_size!

abstract type AbstractSolver end # abstract type for the solvers 

##### Generic functions #####

# iterates through the steps. This is the unsupervised forward pass of the PC network
function forwardSolve!(s::AbstractSolver, x; maxSteps = 50, stoppingCondition = 0.01f0)

    if length(s.errorLogs) != maxSteps
        s.errorLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
        s.duLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
    end

    s.iter_reached = maxSteps #will be updated if the stopping condition is reached before maxSteps

    for i in 1:maxSteps
        forwardSolverStep!(s, x, i)

        if s.duLogs[i] < stoppingCondition 
            s.iter_reached = i
            break
        end
        
    end
end

# iterates through the steps. This is the supervised forward pass of the PC network
function forwardSolve!(s::AbstractSolver, x, y; maxSteps = 50, stoppingCondition = 0.01f0)

    if length(s.errorLogs) != maxSteps
        s.errorLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
        s.duLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
    end

    s.iter_reached = maxSteps #will be updated if the stopping condition is reached before maxSteps

    for i in 1:maxSteps
        forwardSolverStep!(s, x, y, i)

        if s.duLogs[i] < stoppingCondition 
            s.iter_reached = i
            break
        end
        
    end
end

# calculates and logs the values used to analyze the convergence of the forward pass
function forward_convergence_check(s::AbstractSolver, i)
    s.c1 .= abs.(s.pcmodule.errors)

    s.errorLogs[i] = sum(s.c1)
    
    s.c1 .= abs.(s.pcmodule.du)
    s.c2 .= s.pcmodule.u .^ 2
    s.duLogs[i] = dot(s.c1, s.pcmodule.u) / (eps() + sum(s.c2))

end

# calculates and logs the values used to analyze the convergence of the backward pass
function backward_convergence_check(s::AbstractSolver)

    s.pcmodule.errors .= abs.(s.pcmodule.errors)

    # we append because we have no clue how many iterations we will need. Also, appending to an array is dummy fast in Julia
    append!(s.errorLogs, sum(s.pcmodule.errors))
    
    s.c1 .= abs.(s.pcmodule.psgrads) .* abs.(s.pcmodule.ps)
    s.c2 .= s.pcmodule.ps .^ 2
    append!(s.duLogs, sum(s.c1) / (eps() + sum(s.c2)))

end


##### Euler's method #####
mutable struct EulerSolver <: AbstractSolver
    pcmodule
    dt

    c1
    c2
    errorLogs
    duLogs

    iter_reached
    mode
end

function ForwardEulerSolver(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    EulerSolver(m, dt, c1, c2, errorLogs, duLogs, 1, "forward")
end

function BackwardEulerSolver(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    EulerSolver(m, dt, c1, c2, errorLogs, duLogs, 1, "backward")

end

function forwardSolverStep!(s::EulerSolver, x, i)

    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.pcmodule.du, s.pcmodule.u, s.pcmodule, x)
    
    s.pcmodule.u .+= s.dt .* s.pcmodule.du
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)

end

function forwardSolverStep!(s::EulerSolver, x, y, i)

    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.pcmodule.du, s.pcmodule.u, s.pcmodule, x, y)

    s.pcmodule.u .+= s.dt .* s.pcmodule.du
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)

end

function backwardSolverStep!(s::EulerSolver, checkConvergence)
    
    s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
    get_gradient_parameters!(s.pcmodule.psgrads, s.pcmodule)

    s.pcmodule.ps .+= (s.dt / s.pcmodule.nObs) .* s.pcmodule.psgrads
    s.pcmodule.ps.params .= relu.(s.pcmodule.ps.params)
    
    if checkConvergence
        backward_convergence_check(s)
    end

    normalize_receptive_fields!(s.pcmodule)

end

function change_step_size!(s::EulerSolver, stepSize::NamedTuple)
    s.dt = stepSize.dt
end

function PCLayers.change_nObs!(s::EulerSolver, nObs)

    if s.mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end

end

function PCModules.to_gpu!(s::EulerSolver)
    
    if s.mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end

function PCModules.to_cpu!(s::EulerSolver)

    if s.mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end



##### Heun's method #####


mutable struct HeunSolver <: AbstractSolver
    pcmodule
    dt

    k1 # value of u at the start of the step
    dk1 # gradient of u at the start of the step
    
    c1
    c2
    errorLogs
    duLogs

    iter_reached
    mode
end

function ForwardHeunSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.u0)
    dk1 = deepcopy(m.u0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    HeunSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "forward")
end

function BackwardHeunSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.ps)
    dk1 = deepcopy(m.ps)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    HeunSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "backward")

end

function forwardSolverStep!(s::HeunSolver, x, i)

    # first step
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.dk1, s.pcmodule.u, s.pcmodule, x)
    s.k1 .= relu.(s.pcmodule.u .+ ((s.dt / 2) .* s.dk1))

    # second step
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.k1)
    get_gradient_activations!(s.pcmodule.du, s.k1, s.pcmodule, x)
    s.pcmodule.u .+= s.dt .* s.pcmodule.du

    # enforce nonnegativity of activations
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)

end

function forwardSolverStep!(s::HeunSolver, x, y, i)

    # first step
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.dk1, s.pcmodule.u, s.pcmodule, x, y)
    s.k1 .= relu.(s.pcmodule.u .+ ((s.dt / 2) .* s.dk1))

    # second step
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.k1)
    get_gradient_activations!(s.pcmodule.du, s.k1, s.pcmodule, x, y)
    s.pcmodule.u .+= s.dt .* s.pcmodule.du

    # enforce nonnegativity of activations
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)
   

end

function backwardSolverStep!(s::HeunSolver, checkConvergence)
    
    s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
    get_gradient_parameters!(s.dk1, s.pcmodule)
    s.k1 .= s.pcmodule.ps .+ ((s.dt / (2 * s.pcmodule.nObs)) .* s.dk1)

    make_predictions!(s.pcmodule, s.k1.params, s.pcmodule.u)
    get_u0!(s.pcmodule.u0, s.pcmodule, s.k1.initps, s.pcmodule.inputlayer.data)
    s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
    get_gradient_parameters!(s.pcmodule.psgrads, s.pcmodule)

    s.pcmodule.ps .+= (s.dt / s.pcmodule.nObs) .* s.pcmodule.psgrads
    s.pcmodule.ps.params .= relu.(s.pcmodule.ps.params)
    
    if checkConvergence
        backward_convergence_check(s)
    end

    normalize_receptive_fields!(s.pcmodule)

end

function change_step_size!(s::HeunSolver, stepSize::NamedTuple)
    s.dt = stepSize.dt
end

function PCLayers.change_nObs!(s::HeunSolver, nObs)

    if s.mode == "forward"
        s.k1 = deepcopy(s.pcmodule.u0)
        s.dk1 = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.k1 = deepcopy(s.pcmodule.ps)
        s.dk1 = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end

end

function PCModules.to_gpu!(s::HeunSolver)
    
    if s.mode == "forward"
        s.k1 = deepcopy(s.pcmodule.u0)
        s.dk1 = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.k1 = deepcopy(s.pcmodule.ps)
        s.dk1 = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end

function PCModules.to_cpu!(s::HeunSolver)

    if s.mode == "forward"
        s.k1 = deepcopy(s.pcmodule.u0)
        s.dk1 = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.k1 = deepcopy(s.pcmodule.ps)
        s.dk1 = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end




end