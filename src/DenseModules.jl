module DenseModules

#= 
Data structures and functions for dense modules. These are used when one input sample is a vector and scale levels are connected via
dense connections. For training, multiple samples should be included in the input, in which case each sample is a column of a matrix.
=#

########## External Dependencies ##########
using LinearAlgebra, NNlib, ComponentArrays, OrdinaryDiffEq, CUDA


########## Internal Dependencies ##########
#include("./Utils.jl")
#using .Utils

########## Exports ##########
export PCDense, DenseInitializer

########## Data structures ##########

# Dense layer. This is not callable as in Flux layers, but is used as a building block for DenseModules.
mutable struct PCDense

    statesize #size of state array
    ps #Tuple giving the learnable parameters used to predict the layer below
    σ #activation function

    
    grads #Tuple giving gradients of parameters to use in-place operations for weight updates
    ps2 #Tuple giving the parameters squared to use in-place operations for weight normalization
    receptiveFieldNorms #Gives the L2 norm of each receptive field for normalization of weights
    tc #time constant
    α #learning rate
    is_supervised #whether the layer is used for supervised learning
    labels #labels for supervised learning
    name
    T
   
    initializer!
end

# Callable struct that initializes u0 before running the ODE solver with a feedforward densely connected network with ReLU activations.
mutable struct DenseInitializer

    ps
    σ
    grads
    α

end


########## Constructors ##########

# Construct a dense hidden layer.
function PCDense(in_dims, out_dims, name::Symbol, T = Float32; σ = relu, tc = 0.1f0, α = 0.01f0)

    #statesize = zeros(T, out_dims...)
    ps = rand(T, in_dims[1], out_dims[1])
    broadcast!(relu, ps, ps)

    grads = deepcopy(ps)
    ps2 = ps .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1)
    ps ./= receptiveFieldNorms

    is_supervised = false
    labels = [1.0f0]

    ps_initializer = rand(T, in_dims[1], out_dims[1])
    grads_initializer = deepcopy(ps_initializer)
    initializer! = DenseInitializer(ps_initializer, σ, grads_initializer, α)
    
    PCDense(out_dims, ps, σ, grads, ps2, receptiveFieldNorms, tc, α, is_supervised, labels, name, T, initializer!)
end



########## Functions ##########

# Makes the dense layer callable to calculate state updates in the ODE system
function (m::PCDense)(du_l, u_l, predictions_lm1, errors_lm1, errors_l)

    if m.is_supervised
        u_l .= m.labels
        du_l .= zero(eltype(du_l))
        mul!(predictions_lm1, m.ps, u_l)

    else
        broadcast!(m.σ, u_l, u_l)
        du_l .= mul!(du_l, transpose(m.ps), errors_lm1) .- errors_l
        mul!(predictions_lm1, m.ps, u_l)
    end


end


# Makes the dense layer callable to calculate parameter updates in the ODE callback
function (m::PCDense)(errors_lm1, u_l)
   
    m.ps .+= (m.α / size(errors_lm1, 2)) .* mul!(m.grads, errors_lm1, transpose(u_l))

    #broadcast!(relu, m.ps[k], m.ps[k])
    #m.ps2 .= m.ps .^ 2
    #sum!(m.receptiveFieldNorms, m.ps2)
    #m.ps ./= m.receptiveFieldNorms

    #m.initializer!.ps .+= (m.α / size(errors_lm1, 2)) .* mul!(m.initializer!.grads, initerror, transpose(u0))
    
end
    

# Makes the DenseInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::DenseInitializer)(u0, x)
    
    u0 .= m.σ.(mul!(u0, transpose(m.ps), x))
    
end

# Makes the DenseInitializer callable for training. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::DenseInitializer)(initerror_l, u0_lm1, uT_l)

    m.ps .+=  (m.α / size(initerror_l, 2)) .* mul!(m.grads, initerror_l, transpose(u0_lm1))

end



end