import Random
using PyCall
import DSP
import FFTW
import Flux
using TensorCast


function glu(A, dim=-1)
    if dim < 0
        dim = ndims(A) + dim + 1
    end
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

struct S4D
    kernel::S4DKernel
    D::Vector{Float64}
    dropout::Flux.Dropout
    out::Flux.Chain
end

function build_S4D(d_model::Int, d_state::Int=64, dropout::Float64=0., 
                   transposed::Bool=true, kernel_args...)
    D = rand(d_model) 
    kernel = build_S4DKernel(d_model, d_state)
    drop = Flux.Dropout(dropout)
    conv = Flux.Conv((1,), d_model => 2 * d_model)
    return S4D_1(kernel, D, drop, Chain(conv, x -> glu(x, 2)))
end

"""
Input shape u: [L, H, B]
"""
function (s4d::S4D)(u)
    L = size(u, 1)
    H = size(u, 2)
    k = s4d.kernel(H, L)'  # (L H)

    @assert size(u)[1:2] == size(k)

    k_f = FFTW.rfft(cat(k, zeros(size(k)), dims=1), 1)  # (L H)
    u_f = FFTW.rfft(cat(u, zeros(size(u)), dims=1), 1)  # (L H B)
    y = FFTW.irfft(u_f .* k_f, H, 1)[1:L, :, :]  # (L H B)

    y = y + reshape(s4d.D, 1, 10, 1) .* u

    y = s4d.dropout(Flux.gelu(y))  # TODO: replace with DropoutND
    y = glu(conv(y), 2)
    y = s4d.out(y)
    return y
end