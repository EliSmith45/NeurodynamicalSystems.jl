

module PCInputLayers


# External Dependencies
using StatsBase, LinearAlgebra, NNlib, NNlibCUDA, ComponentArrays, CUDA, Reexport

# Internal Dependencies
include("./CommonFunctions.jl")

@reexport import .CommonFunctions

export PCStaticInput 


##### Input layers #####

"""
    PCStaticInput(in_dims, states, name, T)
Input layer who holds the data and does not change over time.
"""
mutable struct PCStaticInput

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
function CommonFunctions.get_gradient_activations!(du, u, layer::PCStaticInput, errors_l)
    du .= 0
    u .= layer.input
end






end