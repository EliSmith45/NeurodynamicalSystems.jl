module PCModules

#=
This module imports and re-exports all predictive coding modules/layers so that any other file can import them all from one place. 
=#

# Files containing any code related to predictive coding modules
include("./InputLayers.jl")
include("./DenseModules.jl")
include("./ConvModules.jl")

using .InputLayers
using .DenseModules
using .ConvModules

# Exported data structures and functions
export PCInput #from InputLayers.jl
export PCDense, DenseModule, DenseInitializer #from DenseModules.jl
export PCConv, ConvModule, ConvInitializer #from ConvModules.jl

end