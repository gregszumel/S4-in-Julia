using Flux: Dense, LayerNorm, Dropout, Chain
using Statistics: mean

include("utils/DenseCustomDim.jl") #DenseCustomDim
include("S4D.jl") #S4D



"""S4 Blocks, which are used for the S4 Model listed below"""
struct S4Block
    s4d::S4D
    norm::LayerNorm
    dropout::Dropout
end

function S4Block(d_model::Int, dropout_p::Float64)
    s4d = S4D(d_model, dropout=dropout_p)
    norm = LayerNorm(d_model)
    dropout = Dropout(dropout_p)
    return S4Block(s4d, norm, dropout)
end


"""Function for permuting dims in s4 forward block"""
fn_permute(X::Array, f) =  permutedims(f(permutedims(X, (2,1,3))), (2,1,3))

"""call function for S4Block"""
function (block::S4Block)(x::Array, prenorm::Bool=false)
    println("Block input size: ", size(x))
    # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
    z = x
    prenorm ?  z = fn_permute(z, block.norm) : nothing
    # Apply S4 block: we ignore the state input and output
    z = block.s4(z)
    # Dropout on the output of the S4 block
    z = block.dropout(z)
    # Residual connection
    x = z .+ x
    !prenorm ?  x = fn_permute(x, block.norm) : nothing
    return x
end

Flux.@functor S4Block

Flux.trainable(m::S4Block) = (m.s4d, m.norm)

# m = S4Block(10, .1) 
# Flux.setup(Adam(), m)

"""
S4 Model. It includes:
    encoder ->  dense layer
    s4blocks -> chain of s4blocks
    decoder ->  dense layer
"""
struct S4Model
    enc::DenseCustomDim
    s4blocks::Chain  # Chain of S4Blocks
    dec::Dense
end

"""Init function for the s4 model"""
function S4Model(d_input::Int, d_output::Int=10, d_model::Int=256, 
                 n_layers::Int=4, dropout_p::Float64=0.2, prenorm::Bool=false)

    # Compact function for building s4 block
    chain = Chain([S4Block(d_model, dropout_p) for _ in 1:n_layers]...)
    encoder = DenseCustomDim(d_input => d_model, 2)
    decoder = Dense(d_model => d_output)
    return S4Model(encoder, chain, decoder)
end


"""
Call function for the S4 Model
Input x is shape (L, d_input, B)
"""
function (m::S4Model)(x::Array)
    println("Model input size: ", size(x))
    x = m.enc(x)  # (L, d_model B)
    println("Model post-encoder size: ", size(x))
    # x = permutedims(x, (2, 1, 3))  # (L, d_model, B)
    x = m.s4blocks(x)   # (L, d_model, B)
    x = mean(x, dims=1)[1, :, :]  # (d_model, B)
    x = m.dec(x)  # (d_output, B)
    return x
end


Flux.@functor S4Model

# m = S4Model(10)
# Flux.setup(Adam(), m)