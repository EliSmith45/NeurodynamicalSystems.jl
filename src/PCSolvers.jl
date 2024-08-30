module PCSolvers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./PCModules.jl")

@reexport using .PCModules

export ES1, forwardES1, backwardES1, forwardSolverStep!, backwardSolverStep!, forwardSolve!

##### Fixed Point Solvers #####

mutable struct ES1
    pcmodule
    dt

    c1
    c2
    errorLogs
    duLogs

    iter_reached
end

function forwardES1(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.u0)
    c2 = deepcopy(m.u0)

    errorLogs = zeros(eltype(m.u0), 1)
    duLogs = zeros(eltype(m.u0), 1)

    ES1(m, dt, c1, c2, errorLogs, duLogs, 1)
end

function backwardES1(m::PCModule; dt = 0.001f0)

    c1 = deepcopy(m.ps)
    c2 = deepcopy(m.ps)

    errorLogs = eltype(m.ps)[]
    duLogs = eltype(m.ps)[]

    ES1(m, dt, c1, c2, errorLogs, duLogs, 1)
end

function PCLayers.change_nObs!(s::ES1, nObs, mode)

    if mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end

end


function forwardSolverStep!(s::ES1, x, i)
    get_gradient_activations!(s.pcmodule, x)
    s.pcmodule.u .+= s.dt .* s.pcmodule.du
    s.pcmodule.u .= relu.(s.pcmodule.u)

    s.c1 .= abs.(s.pcmodule.errors)
    s.errorLogs[i] = sum(s.c1)
    
    s.c1 .= abs.(s.pcmodule.du)
    s.c2 .= s.pcmodule.u .^ 2
    s.duLogs[i] = dot(s.c1, s.pcmodule.u) / (eps() + sum(s.c2))

end

function forwardSolverStep!(s::ES1, x, y, i)
    get_gradient_activations!(s.pcmodule, x, y)
    s.pcmodule.u .+= s.dt .* s.pcmodule.du
    s.pcmodule.u .= relu.(s.pcmodule.u)

    s.c1 .= abs.(s.pcmodule.errors)
    s.errorLogs[i] = sum(s.c1)
    
    s.c1 .= abs.(s.pcmodule.du)
    s.c2 .= s.pcmodule.u .^ 2
    s.duLogs[i] = dot(s.c1, s.pcmodule.u) / (eps() + sum(s.c2))

end

function backwardSolverStep!(s::ES1)
    get_gradient_parameters!(s.pcmodule)
    s.pcmodule.ps .+= s.dt .* s.pcmodule.psgrads
    s.pcmodule.ps.params .= relu.(s.pcmodule.ps.params)
    
    #s.pcmodule.errors .= abs.(s.pcmodule.errors)
    append!(s.errorLogs, sum(abs.(s.pcmodule.errors)))
    
    
    s.c1 .= abs.(s.pcmodule.psgrads)
    s.c2 .= s.pcmodule.ps .^ 2
    append!(s.duLogs, dot(s.c1, s.pcmodule.ps) / (eps() + sum(s.c2)))

    normalize_receptive_fields!(s.pcmodule)

end

function forwardSolve!(s::ES1, x; maxSteps = 50, stoppingCondition = 0.01f0)

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

function forwardSolve!(s::ES1, x, y; maxSteps = 50, stoppingCondition = 0.01f0)

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

function PCModules.to_gpu!(s::ES1, mode)
    if mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end

function PCModules.to_cpu!(s::ES1, mode)
    if mode == "forward"
        s.c1 = deepcopy(s.pcmodule.u0)
        s.c2 = deepcopy(s.pcmodule.u0)
    elseif mode == "backward"
        s.c1 = deepcopy(s.pcmodule.ps)
        s.c2 = deepcopy(s.pcmodule.ps)
    end
    
end



end