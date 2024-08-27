

module PCConvLayers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./CommonFunctions.jl")

@reexport using .CommonFunctions

export PCConv, PCStaticInput  #, change_nObs!

mutable struct PCConv

    statesize #size of state array as a tuple. That is, a dense layer of 64 neurons has a state size of (64,)
    inputsize
    cdims
    σ #activation function

    
    threshold #minimum activation. Each iteration, the threshold is subtracted from du. This greatly accelerates convergence and sets the minimum possible activation.
    name
    T
   
end

function PCConv(k, ch::Pair, inDims, name::Symbol, T = Float32; prop_zero = 0.5, σ = relu, stride=1, padding=0, dilation=1, groups=1, threshold = 0.05f0)

  
    outDims = [inDims...]
    outDims[length(inDims) - 1] = ch[2]
    outDims = Tuple(outDims)
    cdims = NNlib.DenseConvDims(outDims, size(ps); stride = stride, padding = padding, dilation = dilation, groups = groups, flipkernel = true)
    PCConv(outDims, inDims, cdims, σ, threshold, name, T)

    ps = rand(T, k..., ch[2], ch[1])
    nz = Int(round(prop_zero * length(ps)))
    z = sample(1:length(ps), nz, replace = false)
    ps[z] .= 0.0f0
    


    grads = deepcopy(ps)
    ps2 = deepcopy(ps) .^ 2
    receptiveFieldNorms = sum(ps2, dims = 1:(ndims(ps) - 1))
    ps ./= (receptiveFieldNorms .+ eps())
    
    
    states = ∇conv_data(zeros(T, inDims), ps, cdims)
    
    
end

function CommonFunctions.allocate_params(layer::PCConv)
    ps = relu(rand(layer.T, layer.inputsize..., layer.statesize...) .- 0.5f0)
    #nz = Int(round(prop_zero * length(ps)))
    #z = sample(1:length(ps), nz, replace = false)
    #ps[z] .= 0.0f0
    #ps
end
function CommonFunctions.allocate_initparams(layer::PCConv)
    ps = rand(layer.T, layer.statesize..., layer.inputsize...) .- 0.5f0
end

function CommonFunctions.allocate_receptive_field_norms(layer::PCConv)
    zeros(layer.T, 1, layer.statesize...)
end

function CommonFunctions.make_predictions!(predictions_lm1, layer::PCConv, ps, u_l)
    mul!(predictions_lm1, ps, u_l)
    return
end


function CommonFunctions.get_gradient_activations!(du, layer::PCConv, errors_lm1, errors_l, ps)
    du .= mul!(du, transpose(ps), errors_lm1) .- errors_l .- layer.threshold
end

function CommonFunctions.get_gradient_parameters!(grads, layer::PCConv, errors_lm1, u_l)
    mul!(grads, errors_lm1, transpose(u_l))
end

function CommonFunctions.get_gradient_init_parameters!(grads, layer::PCConv, errors_l, u_lm1)
    mul!(grads, errors_l, transpose(u_lm1))
end

function CommonFunctions.get_u0!(u0, layer::PCConv, initps, x)
    u0 .= layer.σ.(mul!(u0, initps, x))
end




end