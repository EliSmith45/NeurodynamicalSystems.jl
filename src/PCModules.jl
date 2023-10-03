module PCModules

#=
This module imports and re-exports all predictive coding modules/layers so that any other file can import them all from one place. 
=#

# Files containing any code related to predictive coding modules
include("./InputLayers.jl")
include("./DenseModules.jl")
include("./ConvModules.jl")
include("./CompositeModules.jl")

using .InputLayers
using .DenseModules
using .ConvModules
using .CompositeModules

# Exported data structures and functions
export PCInput #from InputLayers.jl
export PCDense, CompositeModule, DenseInitializer #from DenseModules.jl
export PCConv, ConvInitializer #from ConvModules.jl


end