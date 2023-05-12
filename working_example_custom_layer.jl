using Flux
using TensorCast

struct S4DKernel
    log_A_real::Matrix
    A_Imag::Matrix
    C_real::Matrix
    C_Imag::Matrix
    log_dt::Vector
end
  
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
  
  # Overload call, so the object can be used as a function
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
  
Flux.@functor S4DKernel
# Flux.trainable(a::Affine) = (a.W, a.b)
m = S4DKernel(10)
Flux.setup(Adam(), m)

Flux.trainable(a::S4DKernel) = (a.A_Imag,)
Flux.setup(Adam(), m)