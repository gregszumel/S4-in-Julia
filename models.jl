using Flux
using Statistics: mean
include("s4d.jl")

simple_conv = Chain(
  Conv((5,5), 3=>16, relu),
  MaxPool((2,2)),
  Conv((5,5), 16=>8, relu),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(200, 120),
  Dense(120, 84),
  Dense(84, 10),
  softmax
)


"""S4 Blocks, which are used for the S4 Model listed below"""
struct S4Block
    s4::S4D
    norm::Flux.LayerNorm
    dropout::Flux.Dropout
end

"""Function for permuting dims in s4 forward block"""
fn_permute(X, f) =  permutedims(f(permutedims(X, (2, 1, 3))), (2, 1, 3))

"""call function for S4Block"""
function (block::S4Block)(x, prenorm=false)
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

"""
S4 Model. It includes:
    encoder ->  dense layer
    s4blocks -> chain of s4blocks
    decoder ->  dense layer
"""
struct S4Model
    encoder::Flux.Dense
    s4blocks::Chain  # Chain of S4Blocks
    decoder::Flux.Dense
end

"""Init function for the s4 model"""
function S4Model(d_input, d_output=10, d_model=256, n_layers=4,
                 dropout_p=0.2, prenorm=false)

    # Compact function for building s4 block
    function init_s4_block(d_model, dropout_p)
        s4d = build_S4D(d_model, dropout=dropout_p)
        norm = Flux.LayerNorm(d_model)
        dropout = Flux.Dropout(dropout_p)
        return S4Block(s4d, norm, dropout)
    end

    chain = Flux.Chain([init_s4_block(d_model, dropout_p) for _ in 1:n_layers]...)
    encoder = Flux.Dense(d_input => d_model)
    decoder = Flux.Dense(d_model => d_output)
    return S4Model(encoder, chain, decoder)
end


"""
Call function for the S4 Model
Input x is shape (d_input, L, B)
"""
function (m::S4Model)(x)
    x = m.encoder(x)  # (d_model, L, B)
    x = permutedims(x, (2, 1, 3))  # (L, d_model, B)
    println(x |> size)
    x = m.s4blocks(x)   # (L, d_model, B)
    x = mean(x, dims=1)[1, :, :]  # (d_model, B)
    x = m.decoder(x)  # (d_output, B)
    return x
end

# m = S4Model(10)
# m(rand(10, 8, 2))