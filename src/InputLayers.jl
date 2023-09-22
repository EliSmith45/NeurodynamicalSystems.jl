module InputLayers

#=

Every PCModule must have an input layer at the lowest scale level to store the input data. This level is held constant throughout the ODE solver
and therefore has no update rules, so the same layer type works for any module. 

=#

########## Exports ##########
export PCInput

########## Data structures ###########

mutable struct PCInput

    states #NamedTuple{:errors, :predictions} giving the values that the layer above predicts for this layer and the prediction error
    name

end


########## Functions ##########

function PCInput(in_dims::Tuple, name::Symbol, T = Float32)
    states = zeros(T, in_dims...)
    PCInput(states, name)
end


end