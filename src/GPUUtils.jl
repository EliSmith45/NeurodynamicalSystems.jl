module GPUUtils


#internal dependencies 
include("./PCModules.jl")
include("./PCNetworks.jl")

using .PCModules
using .PCNetworks

#external dependencies
using CUDA, LinearAlgebra, ComponentArrays

#exports 
#export to_gpu!

########## Functions to move modules to GPU ##########


#=


=#
end