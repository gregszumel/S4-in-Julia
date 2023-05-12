import Random
using PyCall
import DSP
import FFTW
import Flux
using TensorCast

include("utils/GLU.jl") # glu, DropoutNd
include("utils/DropoutNd.jl") # glu, DropoutNd

"""
    S4D Kernel
    Is responsible for constructing the S4D Kernel with hidden dim
    H and sequence length L.
"""
struct S4DKernel
    log_A_real::Matrix
    A_Imag::Matrix
    C_real::Matrix
    C_Imag::Matrix
    log_dt::Vector{Float64}
end


"""Constructing S4D Kernel using fixed parameters"""
function S4DKernel(H::Int, N::Int=64, dt_min::Float64=0.001,
                         dt_max::Float64=0.1, lr::Float64=0.)

    log_dt = rand(H) .* (log(dt_max) - log(dt_min)) .+ log(dt_min)
    C = rand(ComplexF64, H, div(N, 2))
    C_real = real(C)
    C_Imag = imag(C)
    log_A_real = log.(0.5 * ones(H, div(N, 2)))
    A_Imag = pi * (zeros(H, div(N, 2)) .+ transpose(collect(0:div(N, 2)-1)))
    return S4DKernel(log_A_real, A_Imag, C_real, C_Imag, log_dt)
end

"""Forward function of S4DKernel; generates the Kernel of hidden dim H and length L"""
function (k::S4DKernel)(H::Int, L::Int)
    dt = exp.(k.log_dt)
    C = k.C_real + k.C_Imag * 1im
    A = -exp.(k.log_A_real) + k.A_Imag * 1im
    dtA = A .* dt
    K = (ones(size(dtA)..., L) .* dtA ) .* reshape(collect(0:L-1), 1, 1, L)
    C = C .* (exp.(dtA) .- 1.) ./ A
    return_K = zeros(H, L) * 1im
    @reduce return_K[h, l] = sum(n) C[h, n] * exp(K[h, n, l])
    return real.(2 * return_K)
end

Flux.@functor S4DKernel  # what's trainable

# Working example
m = S4DKernel(10)
Flux.setup(Flux.Adam(), m)

"""
    S4D structure and other functions

    Performs the S4D operation with associated kernel, a conv using an FFT,
    and a separate conv with a GLU activation.
"""
struct S4D
    kernel::S4DKernel
    D::Vector{Float64}
    dropout::DropoutNd 
    conv::Flux.Conv
end


"""Builds an S4D model with size d_model (H), d_state (N)"""
function S4D(d_model::Int; d_state::Int=64, dropout::Float64=0., 
                   kernel_args...)
    D = rand(d_model) 
    kernel = S4DKernel(d_model, d_state)
    drop = DropoutNd(dropout)
    conv = Flux.Conv((1,), d_model => 2 * d_model)
    glu_fn(x) = glu(x, 2)
    return S4D(kernel, D, drop, conv)
end

"""
    Forward function for S4D. 

    Input u is an array of shape [L, H, B]
    Output is an array of shape [L, H, B]
"""
function (s4d::S4D)(u::Array)
    L = size(u, 1)
    H = size(u, 2)
    println("L: ", L, ". H: ", H, ".")
    k = s4d.kernel(H, L)'  # (L H)

    @assert size(u)[1:2] == size(k)
    @assert H % 2 == 0

    k_f = FFTW.rfft(cat(k, zeros(size(k)), dims=1), 1)  # (L H)
    u_f = FFTW.rfft(cat(u, zeros(size(u)), dims=1), 1)  # (L H B)
    y = FFTW.irfft(u_f .* k_f, 2*L, 1)[1:L, :, :]  # (L H B)

    y = y + reshape(s4d.D, 1, size(s4d.D, 1), 1) .* u  # (L H B)

    y = s4d.dropout(Flux.gelu(y))  
    y = glu(s4d.conv(y), 2)
    return y
end

Flux.@functor S4D

"""Telling Flux what's trainable in S4D"""
Flux.trainable(s::S4D) = (s.kernel, s.D, s.conv) 

# Working example
m = S4D(10)
Flux.setup(Adam(), m)