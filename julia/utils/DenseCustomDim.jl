using Flux
"""
    Dense Custom Dim(int_out::Pair, dim)

    Builds a Dense Layer and performs the projection on the ~dim~ 
    dimension of the input X

    Ex: 
        X |> size = [5, 3, 2]
        d = DenseCustomDim(3=>7, 2)
        d(X) |> size = [5, 7, 2]
"""

struct DenseCustomDim
    dense::Flux.Dense
    dim::Int
end

DenseCustomDim(in_out::Pair, dim::Int) = DenseCustomDim(Flux.Dense(in_out), dim)

function (d::DenseCustomDim)(X::Array)
    perm = vcat(d.dim,  collect(1:d.dim-1), collect(d.dim+1:ndims(X)))
    unperm = vcat(collect(2:d.dim), 1, collect(d.dim+1:ndims(X)))
    return permutedims(d.dense(permutedims(X, perm)), unperm)
end

Flux.@functor DenseCustomDim

Flux.trainable(m::DenseCustomDim) = (dense=m.dense,)
# d = DenseCustomDim(3=>2, 2)

# Flux.setup(Adam(), d)