module GPUUtils


#internal dependencies 
include("./PCModules.jl")
include("./PCNetworks.jl")

using .PCModules
using .PCNetworks

#external dependencies
using CUDA, LinearAlgebra, ComponentArrays

#exports 
export to_gpu!

########## Functions to move modules to GPU ##########

function to_gpu!(odemodule::DenseModule)

    odemodule.inputstates = cu(odemodule.inputstates)
    odemodule.ps = cu(odemodule.ps)
    odemodule.grads = cu(odemodule.grads)
    odemodule.tc = cu(odemodule.tc)
    odemodule.α = cu(odemodule.α)
    odemodule.u0 = cu(odemodule.u0)
    odemodule.predictions = cu(odemodule.predictions)
    odemodule.errors = cu(odemodule.errors)
   
    to_gpu!(odemodule.initializer!)
end

function to_gpu!(initializer!::DenseInitializer)

    initializer!.ps = cu(initializer!.ps)
    initializer!.grads = cu(initializer!.grads)
    initializer!.α = cu(initializer!.α)
    initializer!.errors = cu(initializer!.errors)
   
end

function to_gpu!(odemodule::ConvModule)

    odemodule.inputstates = cu(odemodule.inputstates)
    odemodule.ps = cu(odemodule.ps)
    odemodule.grads = cu(odemodule.grads)
    odemodule.tc = cu(odemodule.tc)
    odemodule.α = cu(odemodule.α)
    odemodule.u0 = cu(odemodule.u0)
    odemodule.predictions = cu(odemodule.predictions)
    odemodule.errors = cu(odemodule.errors)
   
    to_gpu!(odemodule.initializer!)
end

function to_gpu!(initializer!::ConvInitializer)

    initializer!.ps = cu(initializer!.ps)
    initializer!.grads = cu(initializer!.grads)
    initializer!.α = cu(initializer!.α)
    initializer!.errors = cu(initializer!.errors)
   
end

function to_gpu!(pcn::PCNet)
    to_gpu!(pcn.odemodule)
end

end