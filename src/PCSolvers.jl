##### Fixed Point Solvers #####
module PCSolvers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./PCModules.jl")

@reexport using .PCModules

export EulerSolver, ForwardEulerSolver, BackwardEulerSolver
export HeunSolver, ForwardHeunSolver, BackwardHeunSolver
export EulerAdaptiveSolver, ForwardEulerAdaptiveSolver, BackwardEulerAdaptiveSolver
export Adam, ForwardAdamSolver, BackwardAdamSolver
export Eve, ForwardEveSolver, BackwardEveSolver
export initialize_forward!, initialize_backward!, forwardSolverStep!, backwardSolverStep!, forward_solve!, forward_solve_logged!, change_step_size!

abstract type AbstractSolver end # abstract type for the solvers 

##### Generic functions #####

# iterates through the steps. This is the forward pass of the PC network
function forward_solve!(s::AbstractSolver, x; maxSteps = 50, stoppingCondition = 0.01f0, reinit = true)

    if length(s.errorLogs) != maxSteps
        s.errorLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
        s.duLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
    end

    s.iter_reached = maxSteps #will be updated if the stopping condition is reached before maxSteps

    initialize_forward!(s, x, reinit)

    for i in 1:maxSteps
        forwardSolverStep!(s, x, i)
        
        if s.duLogs[i] < stoppingCondition 
            s.iter_reached = i
            break
        end
        
    end
end




# iterates through the steps of the forward pass and logs the trajectories of the activations
function forward_solve_logged!(s::AbstractSolver, x; maxSteps = 50, stoppingCondition = 0.01f0, reinit = true)

    trajectories = [zeros(eltype(s.pcmodule.u0), s.pcmodule.layers[i].statesize[1:end - 1]..., maxSteps) for i in 1:length(s.pcmodule.layers)]
    
    if length(s.errorLogs) != maxSteps
        s.errorLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
        s.duLogs = zeros(eltype(s.pcmodule.u0), maxSteps)
    end

    s.iter_reached = maxSteps #will be updated if the stopping condition is reached before maxSteps

    initialize_forward!(s, x, reinit)
    

    for i in 1:maxSteps
        forwardSolverStep!(s, x, i)

        for j in eachindex(trajectories)
            trajectories[j][:, i] .= values(NamedTuple(s.pcmodule.u))[j + 1]
        end

        if s.duLogs[i] < stoppingCondition 
            s.iter_reached = i

            for j in eachindex(trajectories)
                trajectories[j] = trajectories[j][:, 1:i]
            end
    
            return trajectories
        end
        
    end

    return trajectories
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

"""
    ForwardEulerSolver(m::PCModule; dt = 0.001f0)

Creates a forward pass solver that uses Euler's method, i.e., traditional gradient descent, to find the optimal activations given the optimal parameters and input.

# Arguments
- `m::PCModule`: The PCmodule.
- `dt::Float32`: The time step size (default: 0.001).

# Examples
"""
function ForwardEulerSolver(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    EulerSolver(m, dt, c1, c2, errorLogs, duLogs, 1, "forward")
end


"""
    BackwardEulerSolver(m::PCModule; dt = 0.001f0)

Creates a backward pass solver that uses Euler's method, i.e., traditional gradient descent, to find the optimal parameters given the optimal activations.


# Arguments
- `m::PCModule`: The PCModule representing the neurodynamical system.
- `dt::Float32`: The time step size for the solver. Default is 0.001.

# Examples
"""
function BackwardEulerSolver(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    EulerSolver(m, dt, c1, c2, errorLogs, duLogs, 1, "backward")

end

#no setup has to be done with Euler's method for the forward pass but we need this function for consistency. 
function initialize_forward!(s::EulerSolver, x, reinit)
    
end

#no setup has to be done with Euler's method for the backward pass but we need this function for consistency. 
function initialize_backward!(s::EulerSolver, reinit)
end


function forwardSolverStep!(s::EulerSolver, x, i)

    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.pcmodule.du, s.pcmodule.u, s.pcmodule, x)
    
    s.pcmodule.u .+= s.dt .* s.pcmodule.du
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)

end


function backwardSolverStep!(s::EulerSolver, checkConvergence, reinit = false)
    
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

"""
    HeunSolver(m::PCModule; dt = 0.001f0)

Creates a solver that uses Heun's method, i.e., a second-order Runge-Kutta method, to find the optimal activations given the optimal parameters and input.

# Arguments
- `m::PCModule`: The PCModule representing the neurodynamical system.
- `dt::Float32`: The time step size for the solver. Default is 0.001.

# Examples
"""
function ForwardHeunSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.u0)
    dk1 = deepcopy(m.u0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    HeunSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "forward")
end

"""
    BackwardHeunSolver(m::PCModule; dt = 0.001f0)

Creates a backward pass solver that uses Heun's method, i.e., a second-order Runge-Kutta method, to find the optimal parameters given the optimal activations.

# Arguments
- `m::PCModule`: The PCModule representing the neurodynamical system.
- `dt::Float32`: The time step size for the solver. Default is 0.001.

# Examples
"""

function BackwardHeunSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.ps)
    dk1 = deepcopy(m.ps)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    HeunSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "backward")

end

function initialize_forward!(s::HeunSolver, x, reinit)
    
    get_u0!(s.pcmodule.u0, s.pcmodule, s.pcmodule.ps.initps, x)
    if reinit
        s.pcmodule.u .= s.pcmodule.u0

        # first step
        make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
        get_gradient_activations!(s.dk1, s.pcmodule.u, s.pcmodule, x)
        s.k1 .= relu.(s.pcmodule.u .+ ((s.dt / 2) .* s.dk1))
        
    end
    
end


function initialize_backward!(s::HeunSolver, reinit)
    
    if reinit
        s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
        get_gradient_parameters!(s.dk1, s.pcmodule)
        s.k1 .= s.pcmodule.ps .+ ((s.dt / (2 * s.pcmodule.nObs)) .* s.dk1)
    end
    
end


function forwardSolverStep!(s::HeunSolver, x, i)

    # second step
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.k1)
    get_gradient_activations!(s.pcmodule.du, s.k1, s.pcmodule, x)
    s.pcmodule.u .+= s.dt .* s.pcmodule.du

    # enforce nonnegativity of activations
    s.pcmodule.u .= relu.(s.pcmodule.u)

    s.dk1 .= s.pcmodule.du
    s.k1 .= relu.(s.pcmodule.u .+ (s.dt .* s.dk1))
    forward_convergence_check(s, i)

end


function backwardSolverStep!(s::HeunSolver, checkConvergence)
    
    
    make_predictions!(s.pcmodule, s.k1.params, s.pcmodule.u)
    get_u0!(s.pcmodule.u0, s.pcmodule, s.k1.initps, s.pcmodule.inputlayer.data)
    s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
    get_gradient_parameters!(s.pcmodule.psgrads, s.pcmodule)

    s.pcmodule.ps .+= (s.dt / s.pcmodule.nObs) .* s.pcmodule.psgrads
    s.pcmodule.ps.params .= relu.(s.pcmodule.ps.params)
    
    s.dk1 .= s.pcmodule.psgrads
    s.k1 .= relu.(s.pcmodule.ps .+ (s.dt / s.pcmodule.nObs) .* s.dk1)

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


##### Euler's method with Runge-Kutta-Feldberg style adaptive stepping #####


mutable struct EulerAdaptiveSolver <: AbstractSolver
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

function ForwardEulerAdaptiveSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.u0)
    dk1 = deepcopy(m.u0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    EulerAdaptiveSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "forward")
end

function BackwardEulerAdaptiveSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.ps)
    dk1 = deepcopy(m.ps)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    EulerAdaptiveSolver(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "backward")

end

##### Adam optimiser #####


mutable struct Adam <: AbstractSolver
    pcmodule
    α
    β1
    β2

    m # value of u at the start of the step
    v # gradient of u at the start of the step
    
    c1
    c2
    errorLogs
    duLogs

    iter_reached
    mode
end

function ForwardAdamSolver(mo::PCModule; α = 0.001f0, β1 = 0.9f0, β2 = 0.999f0)

    m = deepcopy(mo.u0)
    v = deepcopy(mo.u0)

    c1 = deepcopy(mo.u0)
    c2 = deepcopy(mo.u0)

    errorLogs = zeros(eltype(mo.u0), 1)
    duLogs = zeros(eltype(mo.u0), 1)

    Adam(mo, α, β1, β2, m, v, c1, c2, errorLogs, duLogs, 1, "forward")
end

function BackwardAdamSolver(mo::PCModule; α = 0.001f0, β1 = 0.9f0, β2 = 0.999f0)

    m = deepcopy(mo.ps)
    v = deepcopy(mo.ps)

    c1 = deepcopy(mo.ps)
    c2 = deepcopy(mo.ps)

    errorLogs = eltype(mo.ps)[]
    duLogs = eltype(mo.ps)[]

    Adam(mo, α, β1, β2, m, v, c1, c2, errorLogs, duLogs, 1, "backward")

end

#  We must zero out the m and v arrays every time we switch to a new batch with Adam, because the activations have a different fixed point across batches
function initialize_forward!(s::Adam, x, reinit) 
    s.m .= 0
    s.v .= 0
end

# We do NOT zero out m and v on the backward pass, because the parameters have the same fixed point across all batches
function initialize_backward!(s::Adam, reinit)
end


function forwardSolverStep!(s::Adam, x, i)

    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_gradient_activations!(s.pcmodule.du, s.pcmodule.u, s.pcmodule, x)
    
    s.m .= s.β1 .* s.m .+ (1 - s.β1) .* s.pcmodule.du
    s.v .= s.β2 .* s.v .+ (1 - s.β2) .* (s.pcmodule.du .^ 2)
    
    s.m ./= (1 - s.β1 ^ i)
    s.v ./= (1 - s.β2 ^ i)
    
    s.pcmodule.u .+= s.α .* s.m ./ (sqrt.(s.v) .+ eps())
    s.pcmodule.u .= relu.(s.pcmodule.u)

    forward_convergence_check(s, i)

end


function backwardSolverStep!(s::Adam, checkConvergence)
    
    make_predictions!(s.pcmodule, s.pcmodule.ps.params, s.pcmodule.u)
    get_u0!(s.pcmodule.u0, s.pcmodule, s.pcmodule.ps.initps, s.pcmodule.inputlayer.data)
    s.pcmodule.initerror .= s.pcmodule.u .- s.pcmodule.u0
    get_gradient_parameters!(s.pcmodule.psgrads, s.pcmodule)

    s.m .= s.β1 .* s.m .+ (1 - s.β1) .* s.pcmodule.psgrads
    s.v .= s.β2 .* s.v .+ (1 - s.β2) .* (s.pcmodule.psgrads .^ 2)
    
    s.m ./= (1 - s.β1)
    s.v ./= (1 - s.β2)
    
    s.pcmodule.ps .+= s.α .* s.m ./ (sqrt.(s.v) .+ eps())
    s.pcmodule.ps.params .= relu.(s.pcmodule.ps.params)
    
    if checkConvergence
        backward_convergence_check(s)
    end

    normalize_receptive_fields!(s.pcmodule)

end

function change_step_size!(s::Adam, stepSize::NamedTuple)
    s.α = stepSize.α
    s.β1 = stepSize.β1
    s.β2 = stepSize.β2
end

function PCLayers.change_nObs!(s::Adam, nObs)

    if s.mode == "forward"
        s.m = deepcopy(s.pcmodule.u0)
        s.v = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.m = deepcopy(s.pcmodule.ps)
        s.v = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end

end

function PCModules.to_gpu!(s::Adam)
    
    if s.mode == "forward"
        s.m = deepcopy(s.pcmodule.u0)
        s.v = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.m = deepcopy(s.pcmodule.ps)
        s.v = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end

function PCModules.to_cpu!(s::Adam)

    if s.mode == "forward"
        s.m = deepcopy(s.pcmodule.u0)
        s.v = deepcopy(s.pcmodule.u0)
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif s.mode == "backward"
        s.m = deepcopy(s.pcmodule.ps)
        s.v = deepcopy(s.pcmodule.ps)
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end



##### Eve optimiser #####

# Eve is a novel optimiser created by yours truly. Like RKF solvers, it adaptively determines the 
# step size using an estimate of the Taylor Series truncation error estimated by comparing the results 
# of an nth order and n+1 order Runge-Kutta solver. However, like Adam, Eve calculates step sizes 
# pointwise, rather than using a single step size across all states. This modifies the trajectory of 
# the state variables and therefore is not suitable for solving general ODEs. However, it is suitable  
# for convex optimization, as it preserves the fixed point. It performs exceptionally well for poorly 
# conditioned problems by rescaling each variable independently, which effectively makes the problem 
# well conditioned without explicitly preconditioning. 

mutable struct Eve <: AbstractSolver
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

function ForwardEveSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.u0)
    dk1 = deepcopy(m.u0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    Eve(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "forward")
end

function BackwardEveSolver(m::PCModule; dt = 0.001f0)

    k1 = deepcopy(m.ps)
    dk1 = deepcopy(m.ps)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    Eve(m, dt, k1, dk1, c1, c2, errorLogs, duLogs, 1, "backward")

end



end