module Utils

#= 
Miscellaneous utility functions
=#

########## External Dependencies ##########
using NNlib, LinearAlgebra, CUDA

########## Exports ##########
export nonneg_normalized!, gaussian_basis, sample_basis, pick_max!


# Force all elements of an array to be nonnegative and normalize features by their L2 norm. 
# These functions are called during training to normalize weights and enforce nonnegativity, 
# which helps with sparse coding. 

function nonneg_normalized!(weight::AbstractArray{T, 2}) where T
    
    for k in axes(weight, 2)
        weight[:, k] .= relu.(weight[:, k])
        weight[:, k] ./= norm(weight[:, k], 2)
    end
    
end

function nonneg_normalized!(weight::AbstractArray{T, 3}) where T
        
    for k in axes(weight, 3)
        weight[:, :, k] .= relu.(weight[:, :, k])
        weight[:, :, k] ./= norm(weight[:, :, k], 2)
    end

end

function nonneg_normalized!(weight::AbstractArray{T, 4}) where T
        
    for k in axes(weight, 4)
        weight[:, :, :, k] .= relu.(weight[:, :, :, k])
        weight[:, :, :, k] ./= norm(weight[:, :, :, k], 2)
    end

end

function nonneg_normalized!(weight::AbstractArray{T, 5}) where T
        
    for k in axes(weight, 5)
        weight[:, :, :, :, k] .= relu.(weight[:, :, :, :, k])
        weight[:, :, :, :, k] ./= norm(weight[:, :, :, :, k], 2)
    end

end


# Creates a gaussian basis to represent dummy features for toy problems
function gaussian_basis(n, m; basesCenters = (1/n):(1/n):1, binCenters = (1/m):(1/m):1, sigma = 1.0, T = Float32)
    
    w = zeros(T, m, n)
  
    for (j, basis) in enumerate(basesCenters)
        for (i, bin) in enumerate(binCenters)
            w[i, j] = exp(-((basis - bin) / (2 * sigma)) ^ 2); #* (exp(-((centers[end] - centers[j]) / (2 * sigma)) ^ 2) - exp(-((centers[end] - centers[j]) / (2 * sigma)) ^ 2))
        end
    end

    foreach(x -> normalize!(x, 2), eachcol(w))

    w
end

# Generate a dummy data set from a linear combination of features. This is for toy problems to 
# evaluate the sparse codes and learned features.
function sample_basis(basis; nObs = 1, nActive = 2, maxCoherence = .999)

    G = basis' * basis
    n = size(basis, 2)
    m = size(basis, 1)

   
    y = zeros(eltype(basis), n, nObs)
    x = zeros(eltype(basis), m, nObs)
    
    a = 1
  
    
    for t in 1:nObs
        possible = collect(axes(basis, 2))
        #println(possible)
        for i in 1:nActive

            j = rand(possible)
            y[j, t] = rand(.5:.001:1)
            possibleNew = []
            for k in possible
                if G[j, k] < maxCoherence
                    append!(possibleNew, k)
                end
            end

            possible = possibleNew


        end

    end

    return basis * y, y


end


function pick_max!(x; dims = 1)
    ind = argmax(x, dims = dims)
    x .= 0.0f0
    x[ind] .= 1.0f0
    return
end
      
end

