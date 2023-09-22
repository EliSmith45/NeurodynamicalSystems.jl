module GPUKernels

#=
GPU kernels to update all scale levels in parallel for any PC module. These should only be used for relatively small networks
or where the input is only one training sample. That is, they are preferable to use when the network is already trained and is
being used for inference, especially in an online real-time setting. Otherwise, the GPU may run out of memory.
=#

########## Dependencies ##########
using LinearAlgebra, ComponentArrays, CUDA

########## Exports ##########
export componentwise_mul!, componentwise_mul_transpose!, dense_predict_update!, dense_update_weights!

########## Functions ##########

# Componentwise multiplication of arrays. Like NNlib.batched_mul!, but for ComponentArrays.
function componentwise_mul!(c, a, b)
    # Get the indices
    i = ((blockIdx().x - 1) * blockDim().x) + threadIdx().x
    j = ((blockIdx().y - 1) * blockDim().y) + threadIdx().y
    k = ((blockIdx().z - 1) * blockDim().z) + threadIdx().z 
    
    
 
    # Check if indices are within bounds
    if  k <= length(a) && i <= size(a[k], 1) && j <= size(b[k], 2)
        # Perform the matrix multiplication
        c[k][i, j] = 0.0f0
        for w in 1:size(a[k], 2)
            c[k][i, j] += a[k][i, w] * b[k][w, j]
        end
    end
    return
end


# Componentwise multiplication of arrays, but the components of a are transposed. Like NNlib.batched_mul!, but for ComponentArrays.
function componentwise_mul_transpose!(c, a, b)
    # Get the indices
    i = ((blockIdx().x - 1) * blockDim().x) + threadIdx().x
    j = ((blockIdx().y - 1) * blockDim().y) + threadIdx().y
    k = ((blockIdx().z - 1) * blockDim().z) + threadIdx().z 
    
    
 
    # Check if indices are within bounds
    if  k <= length(a) && i <= size(a[k], 2) && j <= size(b[k], 2)
        # Perform the matrix multiplication
        c[k][i, j] = 0.0f0
        for w in 1:size(a[k], 2)
            c[k][i, j] += a[k][w, i] * b[k][w, j]
        end
    end
    return
end


# CUDA kernel to update the predictions and du vectors of each layer in a GPU DenseModule
function dense_predict_update!(predictions, du, u, errors, ps, tc)
    # Get the indices
    i = ((blockIdx().x - 1) * blockDim().x) + threadIdx().x
    j = ((blockIdx().y - 1) * blockDim().y) + threadIdx().y
    k = ((blockIdx().z - 1) * blockDim().z) + threadIdx().z 
    
    
 
    # Check if k index is within bounds
    if  k <= length(ps) 

        # Update predictions for component k by multiplying the kth parameter matrix with its corresponding u matrix (component k + 1, since the input layer has states u but no parameters ps)
        if i <= size(predictions[k], 1) && j <= size(predictions[k], 2) #check if i and j are within bounds for the kth component
            # Perform the matrix multiplication
            predictions[k][i, j] = 0.0f0
            for w in 1:size(ps[k], 2)
                predictions[k][i, j] += ps[k][i, w] * u[k + 1][w, j]
            end

        end

        # Update du for component k + 1 by multiplying the kth parameter matrix with the error matrix of the layer below (component k), then subtracting its own layer's error matrix, and finally multiplying by its time constant tc
        if i <= size(du[k + 1], 1) && j <= size(du[k + 1], 2) #check if i and j are within bounds for the kth component
            
            # Perform the matrix multiplication
            du[k + 1][i, j] = 0.0f0
            for w in 1:size(ps[k], 1)
                du[k + 1][i, j] += ps[k][w, i] * errors[k][w, j]
            end

            du[k + 1][i, j] -= errors[k + 1][i, j] # Subtract error
            du[k + 1][i, j] *= tc[k] # Multiply by time constant

        end

    end
    return
end


# CUDA kernel to update the weights of each layer in a GPU DenseModule
function dense_update_weights!(ps, grads, errors, u, α)
    # Get the indices
    i = ((blockIdx().x - 1) * blockDim().x) + threadIdx().x
    j = ((blockIdx().y - 1) * blockDim().y) + threadIdx().y
    k = ((blockIdx().z - 1) * blockDim().z) + threadIdx().z 
    
    
 
    # Check if k index is within bounds
    if  k <= length(ps) 

        # Update predictions for component k by multiplying the kth parameter matrix with its corresponding u matrix (component k + 1, since the input layer has states u but no parameters ps)
        if i <= size(ps[k], 1) && j <= size(ps[k], 2) #check if i and j are within bounds for the kth component
            # Perform the matrix multiplication
            grads[k][i, j] = 0.0f0
            for w in 1:size(errors[k], 2)
                grads[k][i, j] += errors[k][i, w] * u[k][j, w]
            end

            ps[k][i, j] += α[k] * grads[k][i, j]

            if ps[k][i, j] < 0.0f0
                ps[k][i, j] = 0.0f0
            end
        end

    end
    return
end

end