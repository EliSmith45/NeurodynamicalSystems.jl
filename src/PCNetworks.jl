module PCNetworks

#=
Callable Predictive Coding Network objects. These store any PC module, and can be called on an input to run the corresponding ODE system.
=#

########## External Dependencies ##########
using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, CUDA, Reexport
using Flux: Flux, DataLoader



########## Internal Dependencies ##########
include("./PCModules.jl")
include("./FixedPointSolvers.jl")

@reexport using .PCModules
@reexport using .FixedPointSolvers

########## Exports ##########
export PCNet, reset!, change_kwgs!, train! #from PCNetworks.jl
export ODEIntegrator #from FixedPointSolvers.jl


########## Data structures ###########

# PCNetwork contains a PCModule and ODE solver arguments. Calling it on an input runs the full ODE solver for the chosen time span.
mutable struct PCNet

    pcmodule
    initializer!
    fixedPointSolver

end

########## Functions ##########

# Change the ODE solver arguments
function change_kwgs!(m::PCNet; tspan = (0.0f0, 10.0f0), solver = BS3(), abstol = .01f0, reltol = .01f0, save_everystep = false, save_start = false)

    ode = ODEProblem(m.odemodule, m.odeprob.u0, tspan, Float32[])
    integrator = init(ode, solver, abstol = abstol, reltol = reltol, save_everystep = save_everystep, save_start = save_start);
    m.odeprob = ode
    m.integrator = integrator

end

# Set all states to zero to run the network on a new input
function reset!(pcn::PCNet)

    #pcn.pcmodule.u0 .= zero(eltype(pcn.pcmodule.errors))
    #pcn.odeprob.u0 .= zero(eltype(pcn.odemodule.errors))
    pcn.pcmodule.predictions .= zero(eltype(pcn.pcmodule.errors))
    pcn.pcmodule.errors .= zero(eltype(pcn.pcmodule.errors))
    pcn.initializer!.initerror .= zero(eltype(pcn.pcmodule.errors))

    pcn.fixedPointSolver(true)
    pcn.fixedPointSolver.du .= zero(eltype(pcn.fixedPointSolver.du))
    pcn.fixedPointSolver.integrator.dt = pcn.fixedPointSolver.dt
    
end

function PCModules.change_nObs!(pcn::PCNet, nObs::Int)
    isGPU = pcn.pcmodule.inputlayer.states isa CuArray
    pcn.pcmodule, pcn.initializer! = change_nObs!(pcn.pcmodule, nObs)
   
    if isGPU
        to_gpu!(pcn.pcmodule)
        to_gpu!(pcn.initializer!)
    end
     
    pcn.fixedPointSolver = ODEIntegrator(pcn.pcmodule; tspan = pcn.fixedPointSolver.tspan, solver = pcn.fixedPointSolver.solver, abstol = pcn.fixedPointSolver.abstol, reltol = pcn.fixedPointSolver.reltol, save_everystep = pcn.fixedPointSolver.save_everystep, save_start = pcn.fixedPointSolver.save_start, dt = pcn.fixedPointSolver.dt, adaptive = pcn.fixedPointSolver.adaptive, dtmax = pcn.fixedPointSolver.dtmax, dtmin = pcn.fixedPointSolver.dtmin)
    
end


#=



function change_nObs!(m::PCModule, nObs::Int)
    
    isGPU = m.inputlayer.states isa CuArray

    layers = (m.inputlayer, m.layers...)
    names = map(l -> l.name, layers)
    for i in eachindex(layers)
        newsize = Tuple([layers[i].statesize[1:(end - 1)]..., nObs])
        layers[i].statesize = newsize

    end

    newsize = Tuple([layers[1].statesize[1:(end - 1)]..., nObs])
    layers[1].states = zeros(layers[1].T, newsize)

    errors = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
    predictions = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
    u0 = NamedTuple{names}(map(l -> zeros(l.T, l.statesize), layers))
   
    m.errors = ComponentArray(errors)
    m.predictions = ComponentArray(predictions)
    m.u0 = ComponentArray(u0)
    
    if isGPU
        m.errors = cu(m.errors)
        m.predictions = cu(m.predictions)
        m.u0 = cu(m.u0)
        layers[1].states = cu(layers[1].states)
    end
    layers[end].labels = deepcopy(values(m.u0)[end])
end

=#


# Makes PCNetwork callable for unsupervised learning. This sets the input parameters to x before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}; maxSteps = 100, stoppingCondition = 0.01f0, reset_module = true)
    
    #assign input data
    m.pcmodule.inputlayer.states .= x

    #calculate u0 with initializer network if error is low enough, otherwise initialize with zeros
    m.initializer!(m.pcmodule.u0, x)

    #set u0, predictions, and errors to zero
    if reset_module

        reset!(m)

        for k in eachindex(m.pcmodule.layers)
            m.pcmodule.layers[k].is_supervised = false
        end
        
    end

    
    m.fixedPointSolver(maxSteps, stoppingCondition)
    
end


# Makes PCNetwork callable for supervised learning. This sets the input parameters to x and pins the top level to the targets y before running the ODE system.
function (m::PCNet)(x::Union{Array, CuArray}, y::Union{Array, CuArray}; maxSteps = 100, stoppingCondition = 0.01f0, reset_module = true)
    
    #assign input data
    m.pcmodule.inputlayer.states .= x
    m.pcmodule.layers[end].labels .= y
    #calculate u0 with initializer network if error is low enough, otherwise initialize with zeros
    m.initializer!(m.pcmodule.u0, x)




   #set u0, predictions, and errors to zero
    if reset_module

        reset!(m)

        for k in 1:(length(m.pcmodule.layers) - 1)
            m.pcmodule.layers[k].is_supervised = false
        end
        m.pcmodule.layers[end].is_supervised = true
        
    end

    m.fixedPointSolver(maxSteps, stoppingCondition)

end

function PCModules.to_gpu!(x::PCNet)
    
    to_gpu!(x.pcmodule)
    to_gpu!(x.initializer!)
    to_gpu!(x.fixedPointSolver, x.pcmodule)

end

function PCModules.to_cpu!(x::PCNet)
    
    to_cpu!(x.pcmodule)
    to_cpu!(x.initializer!)
    to_cpu!(x.fixedPointSolver, x.pcmodule)
    
end

function  PCModules.to_gpu!(x::ODEIntegrator, pcmodule)
    
    x.odeproblem = ODEProblem(pcmodule, pcmodule.u0, x.tspan, Float32[])
    x.integrator = init(x.odeproblem, x.solver, abstol = x.abstol, reltol = x.reltol, save_everystep = x.save_everystep, save_start = x.save_start, dt = x.dt, alias_u0 = false)
    x.u = x.integrator.u
    x.du = get_du(x.integrator)
    x.c1 = deepcopy(x.u)
    x.c2 = deepcopy(x.u)

end

function  PCModules.to_cpu!(x::ODEIntegrator, pcmodule)
    
    x.odeproblem = ODEProblem(pcmodule, pcmodule.u0, x.tspan, Float32[])
    x.integrator = init(x.odeproblem, x.solver, abstol = x.abstol, reltol = x.reltol, save_everystep = x.save_everystep, save_start = x.save_start, dt = x.dt, alias_u0 = false)
    x.u = x.integrator.u
    x.du = get_du(x.integrator)
    x.c1 = deepcopy(x.u)
    x.c2 = deepcopy(x.u)
    
end




# Trains the PCNetwork in an unsupervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::Flux.DataLoader{W} ; maxSteps = 50, stoppingCondition = 0.05, maxFollowupSteps = 10, epochs = 1, trainstepsPerBatch = 1, decayLrEvery = 500, lrDecayRate = 0.85f0, show_every = 10, normalize_every = 1000) where W <: Union{Array, CuArray}

    nObs = size(m.pcmodule.inputlayer.states)[end]
    
    if nObs != x.batchsize
        change_nObs!(m, x.batchsize)
    end
    
    initerror = zeros(eltype(m.pcmodule.u0), epochs)
    error = zeros(eltype(m.pcmodule.u0), epochs)
   
    for i in 1:epochs
        for xd in x
          
            m(xd; maxSteps = maxSteps, stoppingCondition = stoppingCondition, reset_module = true) #solve the ODE system 
            
            #if the initializer isn't active yet (due to insufficient training), temporarily activate it to update its weights
            if !m.initializer!.isActive
                m.initializer!.isActive = true
                m.initializer!(m.pcmodule.u0, xd)
                m.initializer!.isActive = false
            end

            m.pcmodule(m.fixedPointSolver.integrator) #update layer weights
            m.initializer!(m.fixedPointSolver.integrator) #update initializer weights

            #takes `trainstepsPerBatch` - 1 extra training updates. Each training step only moves the fixed point slightly, 
            #so we can estimate the new fixed point by just running the ODE system for a few extra steps starting
            #at the old fixed point rather than from u0. This allows us to greatly reduce forward passes during training.
            for j in 1:(trainstepsPerBatch - 1)

                #all is the same as before, except we don't reset the module and take fewer steps
                m(xd; maxSteps = maxFollowupSteps, stoppingCondition = stoppingCondition, reset_module = false);
                
                #if the initializer isn't active yet (due to insufficient training), temporarily activate it to update its weights
                if !m.initializer!.isActive
                    m.initializer!.isActive = true
                    m.initializer!(m.pcmodule.u0, xd)
                    m.initializer!.isActive = false
                end

                m.pcmodule(m.fixedPointSolver.integrator) #update layer weights
                m.initializer!(m.fixedPointSolver.integrator) #update initializer weights

            end


        end

        error[i] = m.fixedPointSolver.errorLogs[end]
        m.fixedPointSolver.c1 .= abs.(m.initializer!.initerror)
        initerror[i] += sum(m.fixedPointSolver.c1)

        #=
        if sum(m.fixedPointSolver.c1) < sum(m.fixedPointSolver.u)
            m.initializer!.isActive = true
        else
            m.initializer!.isActive = false
        end
        =#

        if i % decayLrEvery == 0
            for k in eachindex(m.pcmodule.layers)
                m.pcmodule.layers[k].α *= lrDecayRate
                m.initializer!.initializers[k].α *= lrDecayRate
            end
        end

        #print results
        if i % show_every == 0
            println("Iteration $i absolute error: $(error[i]), init error: $(initerror[i])")
        end
        
        if i % normalize_every == 0
            for k in eachindex(m.pcmodule.layers)
                m.pcmodule.layers[k].ps2 .= m.pcmodule.layers[k].ps .^ 2
                sum!(m.pcmodule.layers[k].receptiveFieldNorms, m.pcmodule.layers[k].ps2)
                m.pcmodule.layers[k].receptiveFieldNorms .= sqrt.(m.pcmodule.layers[k].receptiveFieldNorms)
                m.pcmodule.layers[k].ps ./= (eps() .+ m.pcmodule.layers[k].receptiveFieldNorms)
            end
        end
        
    end
end

# Trains the PCNetwork in a supervised manner. Uses a discrete callback function to pause the ODE solver at the times in stops and update each layers' parameters.
function train!(m::PCNet, x::DataLoader{NamedTuple{(:data, :label), Tuple{W, W}}} , tspan = (0.0f0, 100.0f0); solver = BS3(), epochs = 1, stops = [tspan[1], (tspan[2] - tspan[1]) / 2, tspan[2]], decayLrEvery = 500, lrDecayRate = 0.85f0, show_every = 10, normalize_every = 10000, abstol = 0.01f0, reltol = 0.05f0, save_everystep = false) where W <: Union{Array, CuArray}
    
    nObs = size(m.pcmodule.inputlayer.states)[end]
    
    if nObs != x.batchsize
        change_nObs!(m, x.batchsize)
    end
    
    initerror = zeros(eltype(m.pcmodule.u0), epochs)
    error = zeros(eltype(m.pcmodule.u0), epochs)
   
    for i in 1:epochs
        for (xd, yd) in x
          
            m(xd, yd, maxSteps = maxSteps, stoppingCondition = stoppingCondition, reset_module = true) #solve the ODE system 
            
            #if the initializer isn't active yet (due to insufficient training), temporarily activate it to update its weights
            if !m.initializer!.isActive
                m.initializer!.isActive = true
                m.initializer!(m.pcmodule.u0, xd)
                m.initializer!.isActive = false
            end

            m.pcmodule(m.fixedPointSolver.integrator) #update layer weights
            m.initializer!(m.fixedPointSolver.integrator) #update initializer weights

            #takes `trainstepsPerBatch` - 1 extra training updates. Each training step only moves the fixed point slightly, 
            #so we can estimate the new fixed point by just running the ODE system for a few extra steps starting
            #at the old fixed point rather than from u0. This allows us to greatly reduce forward passes during training.
            for j in 1:(trainstepsPerBatch - 1)

                #all is the same as before, except we don't reset the module and take fewer steps
                m(xd, maxSteps = maxFollowupSteps, stoppingCondition = stoppingCondition, reset_module = false);
                
                #if the initializer isn't active yet (due to insufficient training), temporarily activate it to update its weights
                if !m.initializer!.isActive
                    m.initializer!.isActive = true
                    m.initializer!(m.pcmodule.u0, xd)
                    m.initializer!.isActive = false
                end

                m.pcmodule(m.fixedPointSolver.integrator) #update layer weights
                m.initializer!(m.fixedPointSolver.integrator) #update initializer weights

            end

            error[i] += m.fixedPointSolver.errorLogs[end]
            m.fixedPointSolver.c1 .= abs.(m.initializer!.initerror)
            initerror[i] += sum(m.fixedPointSolver.c1)

            if sum(m.fixedPointSolver.c1) < sum(m.fixedPointSolver.u)
                m.initializer!.isActive = true
            else
                m.initializer!.isActive = false
            end

        end

          
        if i % decayLrEvery == 0
            for k in eachindex(m.pcmodule.layers)
                m.pcmodule.layers[k].α *= lrDecayRate
                m.initializer!.initializers[k].α *= lrDecayRate
            end
        end

        #print results
        if i % show_every == 0
            println("Iteration $i absolute error: $(error[i]), init error: $(initerror[i])")
        end
        
        if i % normalize_every == 0
            for k in eachindex(m.pcmodule.layers)
                m.pcmodule.layers[k].ps2 .= m.pcmodule.layers[k].ps .^ 2
                sum!(m.pcmodule.layers[k].receptiveFieldNorms, m.pcmodule.layers[k].ps2)
                m.pcmodule.layers[k].receptiveFieldNorms .= sqrt.(m.pcmodule.layers[k].receptiveFieldNorms)
                m.pcmodule.layers[k].ps ./= (eps() .+ m.pcmodule.layers[k].receptiveFieldNorms)
            end
        end
        
    end
end

end