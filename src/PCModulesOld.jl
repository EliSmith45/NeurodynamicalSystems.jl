"""
Implements all layer types and PC modules. 
"""
module PCModules

# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, OrdinaryDiffEq, CUDA
 

# Exported data structures and functions
export get_gradient_activations, PCLayer, change_nObs!, to_gpu!, to_cpu!, PCStaticInput, PCDynamicInput, PCDense, DenseInitializer, PCConv, ConvInitializer, PCModule, ModuleInitializer

abstract type PCLayer end

########## Dense layers ##########

"""
    PCDense(out_dims, ps, σ, grads, ps2, receptiveFieldNorms, tc, α, is_supervised, labels, name, T, initializer!)

Dense predictive coding layer who predicts the layer below with:

    xhat = ps * σ(u) 

and updates its states `u` with:

    du = ps' * (x - xhat) - error

Dense layers should only be called by the ODE solver, so the user never needs to call it directly. Just define the 
layer and pass it to a PCModule.

"""
mutable struct PCDense <: PCLayer

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons has a state size of (64,)
    inputsize
    #ps #Tuple giving the learnable parameters used to predict the layer below
    σ #activation function

    
    #grads #Tuple giving gradients oL2 norm of each receptive field for normalization of weights
    #ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    #receptiveFieldNorms #Gives the 
    
    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    is_supervised #whether the layer is used for supervised learning
    name
    T
   
end


"""
    PCDense(in_dims, out_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, tc = 0.1f0, α = 0.01f0)

Construct a dense predictive coding layer.
"""
#=
function PCDense(in_dims, out_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, threshold = 0.05f0)

    #ps = rand(T, in_dims[1], out_dims[1])
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    
    #grads = deepcopy(ps)
    #ps2 = ps .^ 2
    #receptiveFieldNorms = sqrt.(sum(ps2, dims = 1))
    #ps ./= (eps() .+ receptiveFieldNorms)

    is_supervised = false

    ps_initializer = (rand(T, in_dims[1], out_dims[1]) .- 0.5f0) .* prop_zero
    #grads_initializer = deepcopy(ps_initializer)
    #initializer! = DenseInitializer(ps_initializer, σ, grads)
    
    PCDense(out_dims, σ, threshold, is_supervised, name, T)#, initializer!
end
=#

function allocate_params(layer::PCDense)
    ps = rand(layer.T, layer.inputsize..., layer.statesize...)
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    #ps
end
function allocate_initparams(layer::PCDense)
    ps = rand(layer.T, layer.statesize..., layer.inputsize...)
end

function allocate_receptive_field_norms(layer::PCDense)
    zeros(layer.T, layer.statesize...)
end

function make_predictions(predictions_lm1, layer::PCDense, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
end


function get_gradient_activations(du, layer::PCDense, errors_lm1, errors_l, ps)
    mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.threshold
end

function get_gradient_parameters(grads, layer::PCDense, errors_lm1, u_l)
    mul!(grads, errors_lm1, transpose(u_l))
end

function initialize_layer_neural(u0, layer::PCDense, initps, x)
    u0 .= layer.σ.(mul!(u0, initps, x))
end







# Makes the dense layer callable to calculate state updates in the ODE system
function (m::PCDense)(du_l, u_l, predictions_lm1, errors_lm1, errors_l, labels)

    if m.is_supervised
        u_l .= labels
        du_l .= zero(eltype(du_l))
        mul!(predictions_lm1, m.ps, u_l)

    else
        u_l .= m.σ(u_l) 
        du_l .= mul!(du_l, transpose(m.ps), errors_lm1) .- errors_l .- m.threshold
        mul!(predictions_lm1, m.ps, u_l)
    end


end


# Makes the dense layer callable to calculate parameter updates in the ODE callback
function (m::PCDense)(errors_lm1, u_l)
   
    m.ps .+= mul!(m.grads, errors_lm1, transpose(u_l)) 
    m.ps .= relu(m.ps)

end

function to_gpu!(x::PCDense)

   
    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
    x.ps2 = cu(x.ps2)
    x.receptiveFieldNorms = cu(x.receptiveFieldNorms)
 
end

function to_cpu!(x::PCDense)

   
    x.ps = Array(x.ps)
    x.grads = Array(x.grads)
    x.ps2 = Array(x.ps2)
    x.receptiveFieldNorms = Array(x.receptiveFieldNorms)
   
end





########## Input layers ##########

"""
    PCStaticInput(in_dims, states, name, T)
Input layer who holds the data and does not change over time.
"""
mutable struct PCStaticInput <: PCLayer

    statesize
    input 
    name
    T

end

"""
    PCStaticInput(in_dims::Tuple, name::Symbol, T = Float32)
Constructor for PCDynamicInput ayers.
"""
function PCStaticInput(in_dims, name::Symbol, T = Float32)
    input = zeros(T, in_dims..., 1)
    PCStaticInput(in_dims, input, name, T)
end

# Makes the static input layer callable to calculate state updates in the ODE system
function get_gradient_activations(du, u, layer::PCStaticInput, errors_l)
    du .= zero(eltype(du))
    u .= layer.input
end



function (m::PCStaticInput)(du_l, u_l, errors_l)
    du_l .= zero(eltype(du_l))
    u_l .= m.input
end

function to_gpu!(x::PCStaticInput)

    x.states = cu(x.states)

end

function to_cpu!(x::PCStaticInput)

    x.states = Array(x.states)

end


"""
    PCDynamicInput(in_dims, states, name, T)
Input layer who holds the data and changess over time to become more consistent with the predictions (similar to associative memory models like Hopfield Networks).
"""
mutable struct PCDynamicInput <: PCLayer

    statesize
    states #NamedTuple{:errors, :predictions} giving the values that the layer above predicts for this layer and the prediction error
    name
    T

end

"""
    PCDynamicInput(in_dims::Tuple, name::Symbol, T = Float32)
Constructor for PCDynamicInput layers.
"""
function PCDynamicInput(in_dims::Tuple, name::Symbol, T = Float32)
    states = zeros(T, in_dims...)
    PCDynamicInput(in_dims, states, name, T)
end

# Makes the static input layer callable to calculate state updates in the ODE system
function (m::PCDynamicInput)(du_l, errors_l)
    du_l .= .-errors_l
end


function to_gpu!(x::PCDynamicInput)

    x.states = cu(x.states)

end

function to_cpu!(x::PCDynamicInput)

    x.states = Array(x.states)

end






########## PCModules ##########

"""
    PCModule(inputlayer, hiddenlayers, u0, predictions, errors, initerror)

Predictive coding module that holds the input layer, hidden layers, initial states, predictions, errors, and initialization errors. 
The module is callable and passed as the ODE function, calculating du each iteration of the forward pass by iterating through the layers 
and calling each layer on the previous layer's states. The module is also callable on the integrator object of the ODE solver for training.

The user should never need to call a PC module directly. Just define the module and pass it to a PCNet.
"""
mutable struct PCModule

    nObs
    inputlayer
    layers
    #initializers

    predictions
    errors
    u
    du

    u0
    initerror
    ps
    initps
    #labelsct
    
    tc
    α
end



"""
    PCModule(inputlayer, hiddenlayers)

Constructor for predictive coding modules 
"""
function PCModule(inputlayer, hiddenlayers, tc, α)

    nObs = size(inputlayer.input, ndims(inputlayer.input))
    layers = (inputlayer, hiddenlayers...)

    predictions, errors, u, du, u0, initerror = allocate_states(layers, nObs)

    #=
    names = map(l -> l.name, layers)
   
    errors = NamedTuple{names}(map(l -> zeros(l.T, l.statesize, nObs), layers))
    predictions = NamedTuple{names}(map(l -> zeros(l.T, l.statesize, nObs), layers))
    u0 = NamedTuple{names}(map(l -> zeros(l.T, l.statesize, nObs), layers))
    initerror = deepcopy(errors)
    errors = ComponentArray(errors)
    predictions = ComponentArray(predictions)
    u = deepcopy(predictions)
    du = deepcopy(u)

    u0 = ComponentArray(u0)
    initerror = ComponentArray(initerror)
    =#
    
    #labels = deepcopy(u0)

    
    
    names = map(l -> l.name, layers)
    ps = NamedTuple{names[2:end]}(map(l -> allocate_params(l), hiddenlayers))
    initps = NamedTuple{names[2:end]}(map(l -> allocate_initparams(l), hiddenlayers))
    ps = ComponentArray(ps)
    initps = ComponentArray(initps)
    #labels = ComponentArray(labels)
    
    #initializer! = ModuleInitializer(initializers, initerror, false)
    PCModule(nObs, inputlayer, hiddenlayers, predictions, errors, u, du, u0, initerror, ps, initps, tc, α)
end

function allocate_states(layers, nObs)

    names = map(l -> l.name, layers)
    predictions = NamedTuple{names}(map(l -> zeros(l.T, l.statesize..., nObs), layers))
    predictions = ComponentArray(predictions)

    errors = deepcopy(predictions)
    u0 = deepcopy(predictions)
    initerror = deepcopy(predictions)
    u = deepcopy(predictions)
    du = deepcopy(predictions)
    
    return predictions, errors, u, du, u0, initerror
end

function get_gradient_activations(m::PCModule, x)
    
    m.inputlayer.input = x
    get_gradient_activations(values(NamedTuple(m.du))[1], values(NamedTuple(m.u))[1], m.inputlayer, values(NamedTuple(m.errors))[1])

    for k in eachindex(m.layers)
        make_predictions(values(NamedTuple(m.predictions))[k], m.layers[k], values(NamedTuple(m.ps))[k], values(NamedTuple(m.u))[k + 1])
    end

    m.errors .= m.u .- m.predictions

    for k in eachindex(m.layers)
        get_gradient_activations(values(NamedTuple(m.du))[k + 1], m.layers[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.errors))[k + 1], values(NamedTuple(m.ps))[k])
    end

end



# Exported data structures and functions
export EulerSolver

##### Euler's method #####

mutable struct EulerSolver
    u
    du
    dt
end

function EulerSolver(m::PCModule)
    EulerSolver(m.u, m.du, m.tc)
end


function solverStep!(s::EulerSolver)

end
























"""
    DenseInitializer(ps, σ, grads, α)

Dense predictive coding initializer who initializes its corresponding layer activations with:

    u0 = σ(ps * x)

Initializers like this one allow the user to estimate the fixed point of the ODE system with a feedforward network. This can dramatically
speed up convergence of the forward pass. The initializer weights are automatically trained via supervised learning during the backward pass
using the initial activations of the preceding layer as the input and the fixed point of the layer as the target.
    

Dense initializers should only be called by the ODE solver, so the user never needs to call it function directly.
"""

mutable struct DenseInitializer

    ps
    σ
    grads

end




    

# Makes the DenseInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::DenseInitializer)(u0, x)
    
    u0 .= m.σ.(mul!(u0, transpose(m.ps), x))
    
end

# Makes the DenseInitializer callable for training. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::DenseInitializer)(initerror_l, u0_lm1, uT_l)

    m.ps .+=  mul!(m.grads, u0_lm1, transpose(initerror_l))

end

function to_gpu!(x::DenseInitializer)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
   
end

function to_cpu!(x::DenseInitializer)

    x.ps = Array(x.ps)
    x.grads = Array(x.grads)
    
end



########## Convolutional layers ##########
"""
    PCConv(out_dims, ps, σ, grads, ps2, receptiveFieldNorms, tc, α, is_supervised, labels, name, T, initializer!)

Convolutional predictive coding layer who predicts the layer below with a convolution and activation function, then updates the states with a convolution transpose.
It behaves much like a dense layer, but with a convolutional structure.

"""
mutable struct PCConv <: PCLayer

    statesize #size of state array
    ps #Tuple giving the learnable parameters used to predict the layer below
    cdims #Convolution arguments
    σ #activation function

    
    grads #Tuple giving gradients of parameters to use in-place operations for weight updates
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    
    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    is_supervised #whether the layer is used for supervised learning
    name
    T

end



"""
    PCConv(k, ch::Pair, in_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, tc = 1.0f0, α = 0.01f0)

Construct a convolutional predictive coding layer.
"""
function PCConv(k, ch::Pair, in_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, threshold = 0.05f0)

  
    ps = rand(T, k..., ch[2], ch[1])
    nz = Int(round(prop_zero * length(ps)))
    z = sample(1:length(ps), nz, replace = false)
    ps[z] .= 0.0f0
    


    grads = deepcopy(ps)
    ps2 = deepcopy(ps) .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1:(ndims(ps) - 1))
    ps ./= (receptiveFieldNorms .+ eps())
    
    out_dims = [in_dims...]
    out_dims[length(in_dims) - 1] = ch[2]
    out_dims = Tuple(out_dims)
    cdims = NNlib.DenseConvDims(out_dims, size(ps); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    states = ∇conv_data(zeros(T, in_dims), ps, cdims)
    
    is_supervised = false
    
    ps_initializer = rand(T, k..., ch[1], ch[2]) .- 0.5f0
    grads_initializer = deepcopy(ps_initializer)
    cdims_init = NNlib.DenseConvDims(in_dims, size(ps_initializer); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    initializer! = ConvInitializer(ps_initializer, cdims_init, σ, grads_initializer)
   
    PCConv(size(states), ps, cdims, σ, grads, ps2, receptiveFieldNorms, threshold, is_supervised, name, T), initializer!
end


# Makes the convolutional layer callable to calculate state updates in the ODE system
function (m::PCConv)(du_l, u_l, predictions_lm1, errors_lm1, errors_l, labels)

    if m.is_supervised
        u_l .= labels
        du_l .= zero(eltype(du_l))
        conv!(predictions_lm1, u_l, m.ps, m.cdims)

    else
        u_l .= m.σ(u_l) #broadcast!(m.σ, u_l, u_l)
        conv!(predictions_lm1, u_l, m.ps, m.cdims)
        du_l .= conv_data!(du_l, errors_lm1, m.ps, m.cdims) .- errors_l .- m.threshold
    end


end


# Makes the convolutional layer callable to calculate parameter updates in the ODE callback
function (m::PCConv)(errors_lm1, u_l)
   
    m.ps[k] .+= ∇conv_filter!(m.grads, u_l, errors_lm1, m.cdims)
    m.ps .= relu(m.ps)
   
end
    
function to_gpu!(x::PCConv)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
    x.ps2 = cu(x.ps2)
    x.receptiveFieldNorms = cu(x.receptiveFieldNorms)
   
end

function to_cpu!(x::PCConv)

    x.ps = Array(x.ps)
    x.grads = Array(x.grads)
    x.ps2 = Array(x.ps2)
    x.receptiveFieldNorms = Array(x.receptiveFieldNorms)
    
end

"""
    ConvInitializer(ps, cdims, σ, grads, α)

Convolutional predictive coding initializer. Just like dense initializers, but for convolutional layers.
"""
mutable struct ConvInitializer

    ps
    cdims
    σ
    grads

end


# Makes the ConvInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::ConvInitializer)(u0, x)

    u0 .= m.σ(conv!(u0, x, m.ps, m.cdims))

end

# Makes the ConvInitializer callable for training. Sets the initializer!first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::ConvInitializer)(initerror_l, u0_lm1, uT_l)

    m.ps .+= ∇conv_filter!(m.grads, u0_lm1, initerror_l, m.cdims)

end

function to_gpu!(x::ConvInitializer)

    x.ps = cu(x.ps)
    x.grads = cu(x.grads)
   
end

function to_cpu!(x::ConvInitializer)

    x.ps = Array(x.ps)
    x.grads = Array(x.grads)
    
end





########## Conv2Dense layers ##########

"""
    PCConv2Dense(out_dims, ps, σ, grads, ps2, receptiveFieldNorms, tc, α, is_supervised, labels, name, T, initializer!)

Dense predictive coding layer who predicts the layer below with:

    xhat = ps * σ(u) 

and updates its states `u` with:

    du = ps' * (x - xhat) - error

Conv2Dense layers should only be called by the ODE solver, so the user never needs to call it directly. Just define the 
layer and pass it to a PCModule.

"""
mutable struct PCConv2Dense <: PCLayer

    statesize #size of state array
    ps #Tuple giving the learnable parameters used to predict the layer below
    σ #activation function

    predictedFlat
    errorsFlat
    grads #Tuple giving gradients of parameters to use in-place operations for weight updates
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    
    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    is_supervised #whether the layer is used for supervised learning
    name
    T
   
end


"""
    PCConv2Dense(in_dims, out_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, tc = 0.1f0, α = 0.01f0)

Construct a dense predictive coding layer.
"""
function PCConv2Dense(in_dims, out_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, threshold = 0.05f0)

    inputsizeFlat = (prod(in_dims[1:end-1]), in_dims[end])
    ps = rand(T, in_dims[1], out_dims[1])
    nz = Int(round(prop_zero * length(ps)))
    z = sample(1:length(ps), nz, replace = false)
    ps[z] .= 0.0f0
    
    predictedFlat = zeros(T, inputsizeFlat)
    errorsFlat = zeros(T, inputsizeFlat)
    grads = deepcopy(ps)
    ps2 = ps .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1)
    ps ./= receptiveFieldNorms

    is_supervised = false

    ps_initializer = rand(T, in_dims[1], out_dims[1]) .- 0.5f0
    #grads_initializer = deepcopy(ps_initializer)
    initializer! = Conv2DenseInitializer(ps_initializer, inputsizeFlat, σ, grads)
    
    PCConv2Dense(out_dims, ps, σ, predictedFlat, errorsFlat, grads, ps2, receptiveFieldNorms, threshold, is_supervised, name, T), initializer!
end

# Makes the dense layer callable to calculate state updates in the ODE system
function (m::PCConv2Dense)(du_l, u_l, predictions_lm1, errors_lm1, errors_l, labels)

    if m.is_supervised
        u_l .= labels
        du_l .= zero(eltype(du_l))
        predictions_lm1 .= reshape(mul!(m.predictedFlat, m.ps, u_l), size(predictions_lm1))
        
    else
        u_l .= m.σ(u_l) 
        du_l .= mul!(du_l, transpose(m.ps), m.reshape(errors_lm1, size(m.predictedFlat))) .- errors_l .- m.threshold
        predictions_lm1 .= reshape(mul!(m.predictedFlat, m.ps, u_l), size(predictions_lm1))
        
    end

end


# Makes the dense layer callable to calculate parameter updates in the ODE callback
function (m::PCConv2Dense)(errors_lm1, u_l)
   
    m.ps .+= mul!(m.grads, reshape(errors_lm1, size(m.predictedFlat)), transpose(u_l))
    m.ps .= relu(m.ps)

end


"""
    Conv2DenseInitializer(ps, σ, grads, α)

Dense predictive coding initializer who initializes its corresponding layer activations with:

    u0 = σ(ps * x)

Initializers like this one allow the user to estimate the fixed point of the ODE system with a feedforward network. This can dramatically
speed up convergence of the forward pass. The initializer weights are automatically trained via supervised learning during the backward pass
using the initial activations of the preceding layer as the input and the fixed point of the layer as the target.
    

Dense initializers should only be called by the ODE solver, so the user never needs to call it function directly.
"""

mutable struct Conv2DenseInitializer

    ps
    inputsizeFlat
    σ
    grads

end




    

# Makes the DenseInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::Conv2DenseInitializer)(u0, x)
    
    u0 .= m.σ.(mul!(u0, transpose(m.ps), reshape(x, m.inputsizeFlat)))
    
end

# Makes the DenseInitializer callable for training. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::Conv2DenseInitializer)(initerror_l, u0_lm1, uT_l)

    m.ps .+=  mul!(m.grads, reshape(u0_lm1, m.inputsizeFlat), transpose(initerror_l))

end



##### Old PCmodule #####

# Makes the PCModule callable to calculate state updates in the ODE system
function (m::PCModule)(du, u, p, t)
    
    m.errors .= u .- m.predictions

    m.inputlayer(values(NamedTuple(du))[1], values(NamedTuple(u))[1], values(NamedTuple(m.errors))[1])
    for k in eachindex(m.layers)
        m.layers[k](values(NamedTuple(du))[k + 1], values(NamedTuple(u))[k + 1], values(NamedTuple(m.predictions))[k], values(NamedTuple(m.errors))[k], values(NamedTuple(m.errors))[k + 1], values(NamedTuple(m.labels))[k + 1])
    end
    values(NamedTuple(m.predictions))[end] .= values(NamedTuple(u))[end] 
    
end

# Makes PCModule callable on the integrator object of the ODE solver for training. This function updates
# each layer's parameters via a callback function
function (m::PCModule)(integrator)


    for k in eachindex(m.layers)
        m.layers[k](values(NamedTuple(m.errors))[k], values(NamedTuple(integrator.u))[k + 1])
    end

end

function to_gpu!(x::PCModule)

    to_gpu!(x.inputlayer)

    for hl in x.layers
        to_gpu!(hl)
    end

  
    x.u0 = cu(x.u0)
    x.predictions = cu(x.predictions)
    x.errors = cu(x.errors)
    x.labels = cu(x.labels)
   
end

function to_cpu!(x::PCModule)

    to_cpu!(x.inputlayer)

    for hl in x.layers
        to_cpu!(hl)
    end

  
    x.u0 = to_cpu!(x.u0)
    x.predictions = to_cpu!(x.predictions)
    x.errors = to_cpu!(x.errors)
    x.labels = to_cpu!(x.labels)

end



"""
    ModuleInitializer(initializers)

Predictive coding module initializer
"""
mutable struct ModuleInitializer

    initializers
    initerror
    isActive

end

# Makes the PCModule initializer callable to calculate u0 
function (m::ModuleInitializer)(u0, x)
    
    u0 .= zero(eltype(u0))
    values(NamedTuple(u0))[1] .= x

    if m.isActive
        
        for k in eachindex(m.initializers)
            m.initializers[k](values(NamedTuple(u0))[k + 1], values(NamedTuple(u0))[k])
        end
    end
    
end


# Makes PCModule callable on the integrator object of the ODE solver for training. This function updates
# each layer's parameters via a callback function.
function (m::ModuleInitializer)(integrator)

    m.initerror .= integrator.u .- integrator.sol.prob.u0

    for k in eachindex(m.initializers)
        m.initializers[k](values(NamedTuple(m.initerror))[k + 1], values(NamedTuple(integrator.sol.prob.u0))[k], values(NamedTuple(integrator.u))[k + 1])
    end

end

function to_gpu!(x::ModuleInitializer)

    for i in x.initializers
        to_gpu!(i)
    end

    x.initerror = cu(x.initerror)

end

function to_cpu!(x::ModuleInitializer)

    for i in x.initializers
        to_cpu!(i)
    end

    x.initerror = to_cpu!(x.initerror)

end


function to_cpu!(x::ComponentArray)

    names = keys(x)
    x = ComponentArray(NamedTuple{names}(Array.(values(NamedTuple(x)))))

end


function change_nObs!(layer::PCLayer, nObs::Int)
    newsize = Tuple([layer.statesize[1:(end - 1)]..., nObs])
    layer.statesize = newsize
end


function change_nObs!(layer::PCStaticInput, nObs::Int)
    newsize = Tuple([layer.statesize[1:(end - 1)]..., nObs])
    layer.statesize = newsize
    layer.states = zeros(layer.T, newsize)
end

function change_nObs!(layer::PCDynamicInput, nObs::Int)
    newsize = Tuple([layer.statesize[1:(end - 1)]..., nObs])
    layer.statesize = newsize
    layer.states = zeros(layer.T, newsize)
end

function change_nObs!(m::PCModule, nObs::Int)
    change_nObs!(m.inputlayer, nObs)
    for l in m.layers
        change_nObs!(l, nObs)
    end

    m = PCModule(m.inputlayer, m.layers, m.initializers)

end




end