import Random
using PyCall
import DSP
import FFTW
import Flux
using TensorCast

function glu(A, Int::dim=-1)
    if dim < 0
        dim = ndims(A) + dim + 1
    end
    @assert A[dim] % 2 == 2, "glu dim not even"
    halfway_thru_dim = Int(size(A)[dim]/2)
    a = selectdim(A, dim, 1:halfway_thru_dim)
    b = selectdim(A, dim, halfway_thru_dim+1:size(A)[dim])
    return a .* Flux.sigmoid(b)
end


struct S4DKernel
    log_A_real::Matrix
    A_Imag::Matrix
    C_real::Matrix
    C_Imag::Matrix
    log_dt::Vector{Float64}
end


function build_S4DKernel(H::Int, N=64, dt_min=0.001, dt_max=0.1, lr=0.)
    log_dt = rand(H) .* (log(dt_max) - log(dt_min)) .+ log(dt_min)
    C = rand(ComplexF64, H, div(N, 2))
    C_real = real(C)
    C_Imag = imag(C)
    log_A_real = log.(0.5 * ones(H, div(N, 2)))
    A_Imag = pi * (zeros(H, div(N, 2)) .+ transpose(collect(0:div(N, 2)-1)))
    return S4DKernel(log_A_real, A_Imag, C_real, C_Imag, log_dt)
end


function (k::S4DKernel)(H, L)
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

# Flux.@functor S4DKernel

"""DropoutNd structures and accompanying functions"""

struct DropoutNd
    p::Float64
    tie::Bool
end

# simplifying generation
DropoutNd(x::Float64) = DropoutNd(x, true)
# case where inactive, just return X; otherwise run through the dropout
(d::DropoutNd)(X, active) = active && d.p != 0 ? d(X) : X

"""
L, H, B
"""
function (d::DropoutNd)(X)
    mask_shape = size(X)[1:2] 
    mask = rand(mask_shape...) .< (1.0 - d.p)
    X = X .* mask .* (1.0/(1.0-d.p))
    return X
end

# Flux.@functor DropoutNd


struct S4D
    kernel::S4DKernel
    D::Vector{Float64}
    dropout::DropoutNd
    conv::Flux.Conv
    glu
end

function build_S4D(d_model::Int; d_state::Int=64, dropout::Float64=0., 
                   kernel_args...)
    D = rand(d_model) 
    kernel = build_S4DKernel(d_model, d_state)
    drop = DropoutNd(dropout)
    conv = Flux.Conv((1,), d_model => 2 * d_model)
    glu_fn(x) = glu(x, 2)
    return S4D(kernel, D, drop, conv, glu_fn)
end

"""
Input shape u: [L, H, B]
"""
function (s4d::S4D)(u)
    L = size(u, 1)
    H = size(u, 2)
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

# Flux.@functor 