module PCNetworks
# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport, Flux

# Internal Dependencies
include("./PCSolvers.jl")
include("./Utils.jl")

@reexport using .PCSolvers
using .Utils

export PCNetwork, log_trajectories, train_unsupervised!, train_supervised!, change_step_size_forward!, recommend_batch_size, change_step_size_backward!
export get_states, get_errors, get_predictions, get_error_logs, get_du_logs, get, get_training_error_logs, get_training_du_logs, get_u0, get_iters, get_model_parameters, get_initializer_parameters



"""
    PCNetwork(pcmodule, actOpt, psOpt)

Callable struct containing a PCModule object, a solver object for the forward pass, and a solver object for the backward pass. The PCNetwork object is callable on the input data (and optionally the labels as well for supervised learning), which will run the forward pass solver until convergence. 
# Fields
- `pcmodule`: A PCModule object.
- `actOpt`: The solver object for the forward pass.
- `psOpt`: The solver object for the backward pass.

# Examples
"""

mutable struct PCNetwork
    pcmodule
    actOpt
    psOpt
end


function (m::PCNetwork)(x; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    if m.pcmodule.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.pcmodule.u0, m.pcmodule, m.pcmodule.ps.initps, x)

    
    if use_neural_initializer
        m.pcmodule.u .= m.pcmodule.u0
    elseif reset_states
        m.pcmodule.u .= 0
        values(NamedTuple(m.pcmodule.u))[1] .= x
    end
    
    forwardSolve!(m.actOpt, x; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end

function (m::PCNetwork)(x, y; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    if m.pcmodule.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.pcmodule.u0, m.pcmodule, m.pcmodule.ps.initps, x, y)

    

    
    if use_neural_initializer
        m.pcmodule.u .= m.pcmodule.u0
    elseif reset_states
        m.pcmodule.u .= 0
        values(NamedTuple(m.pcmodule.u))[1] .= x
    end
    
    forwardSolve!(m.actOpt, x, y; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end

function log_trajectories(m::PCNetwork, x; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    
    if size(x)[end] != 1
        println("This function is only for logging the trajectories of a single observation.")
        return
    end
    
    if m.pcmodule.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.pcmodule.u0, m.pcmodule, m.pcmodule.ps.initps, x)

    
    if use_neural_initializer
        m.pcmodule.u .= m.pcmodule.u0
    elseif reset_states
        m.pcmodule.u .= 0
        values(NamedTuple(m.pcmodule.u))[1] .= x
    end
    
    forward_solve_logged!(m.actOpt, x; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end


function log_trajectories(m::PCNetwork, x, y; maxIters = 50, stoppingCondition = 0.01f0, use_neural_initializer = false, reset_states = true)
    
    if size(x)[end] != 1
        println("This function is only for logging the trajectories of a single observation.")
        return
    end
    
    if m.pcmodule.nObs != size(x)[end]
        change_nObs!(m, size(x)[end])
    end

    get_u0!(m.pcmodule.u0, m.pcmodule, m.pcmodule.ps.initps, x)

    
    if use_neural_initializer
        m.pcmodule.u .= m.pcmodule.u0
    elseif reset_states
        m.pcmodule.u .= 0
        values(NamedTuple(m.pcmodule.u))[1] .= x
    end
    
    forward_solve_logged!(m.actOpt, x, y; maxSteps = maxIters, stoppingCondition = stoppingCondition)
end

"""
    train_unsupervised!(m::PCNetwork, trainingData::Flux.DataLoader; maxIters = 50, stoppingCondition = 0.01f0, epochs = 100, followUpRuns = 10, maxFollowUpIters = 10, print_batch_error = true)

Train the PCNetwork `m` using unsupervised learning on the `trainingData` DataLoader. This learns the parameters which minimize the prediction error across all layers.

# Arguments
- `m::PCNetwork`: The PCNetwork model to be trained, on the GPU if desired.
- `trainingData::Flux.DataLoader`: The DataLoader containing the training data. The DataLoader should be on the CPU, as the batches will be transferred to the GPU automatically. Choose the largest batch size possible without making your GPU panic. 
- `maxIters::Int = 50`: The maximum number of iterations for the forward pass.
- `stoppingCondition::Float32 = 0.01f0`: Forward pass stopping condition, given as the norm of the update as a fraction of the norm of the state.
- `epochs::Int = 100`: The number of epochs for training.
- `followUpRuns::Int = 10`: After a learning step, one can resume the forward pass where it left off rather than restarting with a new batch. This greatly accelerates training and minimizes transfers to the GPU. This variable is the number of times to resume the forward pass and take another training step before moving to the next batch.
- `maxFollowUpIters::Int = 10`: The maximum number of forward pass iterations for each follow-up run.
- `print_batch_error::Bool = true`: Whether to print the batch error during training.

# Examples
"""

function train_unsupervised!(m::PCNetwork, trainingData::Flux.DataLoader; maxIters = 50, stoppingCondition = 0.01f0, epochs = 100, followUpRuns = 10, maxFollowUpIters = 10, print_batch_error = true)
    
    if first(trainingData) isa CuArray

        println("To improve performance, leave the DataLoader on the CPU and make the batch size as large as possible without making your GPU kill itself. Batches will automatically be transferred to the GPU. Increase followUpRuns to take lots of training steps on the same batch to minimize transfers to the GPU.")
        return

    end

    onGPU = (get_states(m)[1:2] isa CuArray) 
    k = 0
    for i in 1:epochs
        
        for xb in trainingData
            
            k += 1
            if onGPU
                xb = cu(xb)
            end

            m(xb; maxIters = maxIters, stoppingCondition = stoppingCondition, use_neural_initializer = true, reset_states = false)
            backwardSolverStep!(m.psOpt, true)
            
            if print_batch_error
                println("Batch $k error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end])")
            end
            
            for j in 1:followUpRuns

                m(xb; maxIters = maxFollowUpIters, stoppingCondition = stoppingCondition, use_neural_initializer = false, reset_states = false)
                backwardSolverStep!(m.psOpt, false)
            end

            if onGPU
                CUDA.unsafe_free!(xb)
            end

        end
        
        
    
                
        println("Epoch $i error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end])")

    end

end


"""
    train_supervised!(m::PCNetwork, trainingData::Flux.DataLoader; maxIters = 50, stoppingCondition = 0.01f0, epochs = 100, followUpRuns = 10, maxFollowUpIters = 10, print_batch_error = true)

Train a PCNetwork model using supervised learning. This still learns to minimize the prediction error across all layers rather than directly learning to predict labels as is typical in supervised learning. Instead, we simply pin the final layer to the labels and learn the parameters which minimize the prediction error.

# Arguments
- `m::PCNetwork`: The PCNetwork model to be trained.
- `trainingData::Flux.DataLoader`: The training data as a Flux DataLoader.
- `maxIters::Int = 50`: The maximum number of iterations for training.
- `stoppingCondition::Float32 = 0.01f0`: The stopping condition for training.
- `epochs::Int = 100`: The number of epochs for training.
- `followUpRuns::Int = 10`: The number of follow-up runs after training.
- `maxFollowUpIters::Int = 10`: The maximum number of iterations for follow-up runs.
- `print_batch_error::Bool = true`: Whether to print the batch error during training.

# Examples
"""

function train_supervised!(m::PCNetwork, trainingData::Flux.DataLoader; maxIters = 50, stoppingCondition = 0.01f0, epochs = 100, followUpRuns = 10, maxFollowUpIters = 10, print_batch_error = true) 
    
    if first(trainingData.data) isa CuArray

        println("To improve performance, leave the DataLoader on the CPU and make the batch size as large as possible without making your GPU kill itself. Batches will automatically be transferred to the GPU. Increase followUpRuns to take lots of training steps on the same batch to minimize transfers to the GPU.")
        return

    end
    
    onGPU = (get_states(m)[1:2] isa CuArray) 
    k = 0
    a = 0
    for i in 1:epochs
   
        for (xb, yb) in trainingData
            k += 1
            if onGPU
                xbc = cu(xb)
                ybc = cu(yb)
            end
            
            m(xbc, ybc; maxIters = maxIters, stoppingCondition = stoppingCondition, use_neural_initializer = true, reset_states = false)
            
            backwardSolverStep!(m.psOpt, true)


            for j in 1:followUpRuns

                m(xbc, ybc; maxIters = maxFollowUpIters, stoppingCondition = stoppingCondition, use_neural_initializer = false, reset_states = false)
                backwardSolverStep!(m.psOpt, false)

            end

            if print_batch_error
                m(xbc; maxIters = maxIters, stoppingCondition = stoppingCondition, use_neural_initializer = true, reset_states = true)
                a = classification_accuracy(get_states(m), ybc)
                println("Batch $k error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end]), accuracy = $a")
            end

            if onGPU
                CUDA.unsafe_free!(xbc)
                CUDA.unsafe_free!(ybc)
            end

        end
        

        println("Epoch $i error: $(m.psOpt.errorLogs[end]), du: $(m.psOpt.duLogs[end])")

    end

end


"""
    recommend_batch_size(n::PCNetwork, memsizeGB)

Recommend the batch size for a given `PCNetwork` based on the available memory size in gigabytes.

# Arguments
- `n::PCNetwork`: The PCNetwork object for which to recommend the batch size.
- `memsizeGB`: The available memory size in gigabytes.

# Returns
- `batch_size::Int`: The recommended batch size.

"""
function recommend_batch_size(n::PCNetwork, memsizeGB)
    memsizeGB *= 1000000000
    dtSize = sizeof(n.pcmodule.inputlayer.T) 
    bytesPerObservation = (length(n.pcmodule.u0) * 7) * dtSize
    bytesForParams = length(n.pcmodule.ps) * 4 * dtSize

    bytesFree = memsizeGB - bytesForParams
    obsFree = bytesFree / bytesPerObservation
    obsFree
end


"""
    change_nObs!(p::PCNetwork, nObs)

Change the number of observations in a PCNetwork and reallocate the appropriately sized state and cache arrays. This is automatically called when the size of the input doesn't line up with the number of observations the network is expecting, so it shouldn't be called manually. Also, reallocating is slow, so it is best to avoid using this as much as possible during training by keeping the dimensionality of each batch the same during training. 

# Arguments
- `p::PCNetwork`: The PCNetwork object.
- `nObs`: The new number of observations.

"""
function PCLayers.change_nObs!(p::PCNetwork, nObs)
    change_nObs!(p.pcmodule, nObs)
    change_nObs!(p.actOpt, nObs)
    change_nObs!(p.psOpt, nObs)
end


"""
    to_gpu!(p::PCNetwork)

Converts the PCNetwork `p` to GPU memory.

# Arguments
- `p::PCNetwork`: The PCNetwork to be converted to GPU memory.

# Examples
"""
function PCModules.to_gpu!(p::PCNetwork)
    to_gpu!(p.pcmodule)
    to_gpu!(p.actOpt)
    to_gpu!(p.psOpt)
end


"""
    to_cpu!(p::PCNetwork)

Move the PCNetwork `p` to the CPU memory.

# Arguments
- `p::PCNetwork`: The PCNetwork object to be moved to CPU memory.

# Examples
"""
function PCModules.to_cpu!(p::PCNetwork)
    to_cpu!(p.pcmodule)
    to_cpu!(p.actOpt)
    to_cpu!(p.psOpt)
end


"""
    change_step_size_forward!(p::PCNetwork, step_size)

Change the step size or other hyperparameters for the forward solver.

# Arguments
- `p::PCNetwork`: The PCNetwork object.
- `step_size::NamedTuple`: The new step size for forward propagation.

"""
function change_step_size_forward!(p::PCNetwork, step_size::NamedTuple)
    change_step_size!(p.actOpt, step_size)
end


"""
    change_step_size_backward!(p::PCNetwork, step_size::NamedTuple)

Change the step size or other hyperparameters for the backward solver.

# Arguments
- `p::PCNetwork`: The PCNetwork object to modify.
- `step_size::NamedTuple`: The new step size to set.

"""
function change_step_size_backward!(p::PCNetwork, step_size::NamedTuple)
    change_step_size!(p.psOpt, step_size)
end


function get_states(p::PCNetwork)
    return p.pcmodule.u
end

function get_errors(p::PCNetwork)
    return p.pcmodule.errors
end

function get_predictions(p::PCNetwork)
    return p.pcmodule.predictions
end

function get_error_logs(p::PCNetwork)
    return p.actOpt.errorLogs[1:p.actOpt.iter_reached]
end

function get_du_logs(p::PCNetwork)
    return p.actOpt.duLogs[1:p.actOpt.iter_reached]
end

function get_training_error_logs(p::PCNetwork)
    return p.psOpt.errorLogs
end

function get_training_du_logs(p::PCNetwork)
    return p.psOpt.duLogs
end

function get_u0(p::PCNetwork)
    return p.pcmodule.u0
end

function get_model_parameters(p::PCNetwork)
    return p.pcmodule.ps.params
end

function get_initializer_parameters(p::PCNetwork)
    return p.pcmodule.ps.initps
end

function get_iters(p::PCNetwork)
    return p.actOpt.iter_reached
end

end