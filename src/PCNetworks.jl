module PCNetworks
# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport, Flux

# Internal Dependencies
include("./PCSolvers.jl")

@reexport using .PCSolvers


export Pnet, trainSteps!, change_step_size_forward!, change_step_size_backward!
export get_states, get_errors, get_predictions, get_error_logs, get_du_logs, get, get_training_error_logs, get_training_du_logs, get_u0, get_iters, get_model_parameters, get_initializer_parameters

mutable struct Pnet
    mo
    actOpt
    psOpt
end

function (m::Pnet)(x; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    if m.mo.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.mo.u0, m.mo, m.mo.ps.initps, x)

    
    if use_neural_initializer
        m.mo.u .= m.mo.u0
    elseif reset_states
        m.mo.u .= 0
        values(NamedTuple(m.mo.u))[1] .= x
    end
    
    forwardSolve!(m.actOpt, x; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end

function (m::Pnet)(x, y; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    if m.mo.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.mo.u0, m.mo, m.mo.ps.initps, x, y)

    

    
    if use_neural_initializer
        m.mo.u .= m.mo.u0
    elseif reset_states
        m.mo.u .= 0
        values(NamedTuple(m.mo.u))[1] .= x
    end
    
    forwardSolve!(m.actOpt, x, y; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end

function trainSteps!(m::Pnet, trainingData::Flux.DataLoader{W}; maxIters = 50, stoppingCondition = 0.01f0, trainingSteps = 100, followUpRuns = 10, maxFollowUpIters = 10) where W <: Union{Array, CuArray}
    
    
    for i in 1:trainingSteps
        for x in trainingData
            
            m(x; maxIters = maxIters, stoppingCondition = stoppingCondition, use_neural_initializer = true, reset_states = false)
            backwardSolverStep!(m.psOpt, true)

            for j in 1:followUpRuns

                m(x; maxIters = maxFollowUpIters, stoppingCondition = stoppingCondition, use_neural_initializer = false, reset_states = false)
                backwardSolverStep!(m.psOpt, false)
            end

        end
            
        println("Epoch $i error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end])")

    end

end


function trainSteps!(m::Pnet, trainingData::Flux.DataLoader{NamedTuple{(:data, :label), Tuple{W, W}}}; maxIters = 50, stoppingCondition = 0.01f0, trainingSteps = 100, followUpRuns = 10, maxFollowUpIters = 10) where W <: Union{Array, CuArray}
    
    
    for i in 1:trainingSteps
        for (x, y) in trainingData
            
            m(x, y; maxIters = maxIters, stoppingCondition = stoppingCondition, use_neural_initializer = true, reset_states = false)
            backwardSolverStep!(m.psOpt, true)

            for j in 1:followUpRuns

                m(x, y; maxIters = maxFollowUpIters, stoppingCondition = stoppingCondition, use_neural_initializer = false, reset_states = false)
                backwardSolverStep!(m.psOpt, false)

            end

        end

        println("Epoch $i error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end])")

    end

end

function PCLayers.change_nObs!(p::Pnet, nObs)
    change_nObs!(p.mo, nObs)
    change_nObs!(p.actOpt, nObs)
    change_nObs!(p.psOpt, nObs)
end

function PCModules.to_gpu!(p::Pnet)
    to_gpu!(p.mo)
    to_gpu!(p.actOpt)
    to_gpu!(p.psOpt)
end

function PCModules.to_cpu!(p::Pnet)
    to_cpu!(p.mo)
    to_cpu!(p.actOpt)
    to_cpu!(p.psOpt)
end

function change_step_size_forward!(p::Pnet, step_size)
    change_step_size!(p.actOpt, step_size)
end

function change_step_size_backward!(p::Pnet, step_size)
    change_step_size!(p.psOpt, step_size)
end

function get_states(p::Pnet)
    return p.mo.u
end

function get_errors(p::Pnet)
    return p.mo.errors
end

function get_predictions(p::Pnet)
    return p.mo.predictions
end

function get_error_logs(p::Pnet)
    return p.actOpt.errorLogs[1:p.actOpt.iter_reached]
end

function get_du_logs(p::Pnet)
    return p.actOpt.duLogs[1:p.actOpt.iter_reached]
end

function get_training_error_logs(p::Pnet)
    return p.psOpt.errorLogs
end

function get_training_du_logs(p::Pnet)
    return p.psOpt.duLogs
end

function get_u0(p::Pnet)
    return p.mo.u0
end

function get_model_parameters(p::Pnet)
    return p.mo.ps.params
end

function get_initializer_parameters(p::Pnet)
    return p.mo.ps.initps
end

function get_iters(p::Pnet)
    return p.actOpt.iter_reached
end

end