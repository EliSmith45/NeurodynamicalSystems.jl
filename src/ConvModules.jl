module ConvModules

#= 
Data structures and functions for convolutional modules. These are used when the input is a structured array like an image or frequency spectrum.
Scale levels are connected via convolutions.
=#

########## External Dependencies ##########
using StatsBase, LinearAlgebra, ComponentArrays, OrdinaryDiffEq, CUDA
using NNlib, NNlibCUDA
#import NNlibCUDA: conv!, ∇conv_data!, ∇conv_filter!

########## Internal Dependencies ##########
include("./Utils.jl")
using .Utils

########## Exports ##########
export PCConv, ConvInitializer

########## Data structures ##########

# Convolutional layer. This is not callable as in Flux layers, but is used as a building block for ConvModules.
mutable struct PCConv

    statesize #size of state array
    ps #Tuple giving the learnable parameters used to predict the layer below
    cdims #Convolution arguments
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

# Callable struct that initializes u0 before running the ODE solver with a feedforward convolutionally connected network with ReLU activations.
mutable struct ConvInitializer

    ps
    cdims
    σ
    grads
    α

end


########## Constructors ##########

# Construct a convolutional hidden layer.
function PCConv(k, ch::Pair, in_dims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, tc = 1.0f0, α = 0.01f0)

  
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
    labels = [false]

    ps_initializer = rand(T, k..., ch[1], ch[2]) .- 0.5f0
    grads_initializer = deepcopy(ps_initializer)
    cdims_init = NNlib.DenseConvDims(in_dims, size(ps_initializer); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    initializer! = ConvInitializer(ps_initializer, cdims_init, σ, grads_initializer, α)
   
    PCConv(size(states), ps, cdims, σ, grads, ps2, receptiveFieldNorms, tc, α, is_supervised, labels, name, T, initializer!)
end



########## Functions ##########


# Makes the convolutional layer callable to calculate state updates in the ODE system
function (m::PCConv)(du_l, u_l, predictions_lm1, errors_lm1, errors_l)

    if m.is_supervised
        u_l .= m.labels
        du_l .= zero(eltype(du_l))
        conv!(predictions_lm1, u_l, m.ps, m.cdims)

    else
        u_l .= m.σ(u_l) #broadcast!(m.σ, u_l, u_l)
        conv!(predictions_lm1, u_l, m.ps, m.cdims)
        du_l .= ∇conv_data!(du_l, errors_lm1, m.ps, m.cdims) .- errors_l
    end


end


# Makes the dense layer callable to calculate parameter updates in the ODE callback
function (m::PCConv)(errors_lm1, u_l)
   
    m.ps[k] .+= ((m.α * size(u_l, ndims(u_l) - 1)) / prod(size(errors_lm1))) .* ∇conv_filter!(m.grads, errors_lm1, u_l, m.cdims)
    m.ps .= relu(m.ps)
   
end
    

# Makes the DenseInitializer callable. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::ConvInitializer)(u0, x)

    u0 .= m.σ(conv!(u0, x, m.ps, m.cdims))

end

# Makes the DenseInitializer callable for training. Sets the first scale level of u0 equal to the input x, then computes the initial values of each scale level
# using the learnable feedforward network parameterized by ps.
function (m::ConvInitializer)(initerror_l, u0_lm1, uT_l)

    m.ps .+= ((m.α * size(u_l, ndims(u_l) - 1)) / prod(size(errors_lm1))) .* ∇conv_filter!(m.grads, u0_lm1, initerror_l, m.cdims)

end



end