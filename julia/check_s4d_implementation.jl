using PyCall

get_param_vals(parameter_tensor)  = parameter_tensor.data.numpy()

function get_s4d_params(m)
    return Dict(name => get_param_vals(param) 
                for (name, param) in m.named_parameters())
end

# Run the original python code for S4D, pulled from the s4d.py file
py"""
import sys
sys.path.insert(0, "/Users/gregszumel/Documents/coding/s4")

import s4d
import torch
s4d_py = s4d.S4D(10)
u = torch.rand(2, 10, 5)
if not s4d_py.transposed: u = u.transpose(-1, -2)
L = u.size(-1)
# Compute SSM Kernel
k = s4d_py.kernel(L=L) # (H L)
# Convolution
k_f = torch.fft.rfft(k, n=2*L) # (H L)
u_f = torch.fft.rfft(u, n=2*L) # (B H L)
y_1 = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)
# Compute D term in state space equation - essentially a skip connection
ud = u * s4d_py.D.unsqueeze(-1)
y_2 = y_1 + ud
y_3 = s4d_py.dropout(s4d_py.activation(y_2))
y_4 = s4d_py.output_linear(y_3)
if not s4d_py.transposed: 
    y_5 = y_4.transpose(-1, -2)
else:
    y_5 = y_4
"""

# build equivalent julia s4d structure, same params
d = get_s4d_params(py"s4d_py")
S4D_jl = S4D(
    S4DKernel(d["kernel.log_A_real"], 
            d["kernel.A_imag"], 
            d["kernel.C"][:, :, 1], 
            d["kernel.C"][:, :, 2], 
            d["kernel.log_dt"]),
    d["D"],
    Flux.Dropout(.1),
    Flux.Chain(
        Flux.Conv(permutedims(d["output_linear.0.weight"], (3, 2, 1)), 
                d["output_linear.0.bias"]),
        x -> glu(x, 2)
    )
)


# grab the same input
u = permutedims(py"u.numpy()", (3, 2, 1))

# run the replicated julia implementation
L = size(u, 1)
H = size(u, 2)
k = S4D_jl.kernel(H, L)'  # (L H)

@assert size(u)[1:2] == size(k)

k_f = FFTW.rfft(cat(k, zeros(size(k)), dims=1), 1)  # (L H)
u_f = FFTW.rfft(cat(u, zeros(size(u)), dims=1), 1)  # (L H B)
y_1 = FFTW.irfft(u_f .* k_f, H, 1)[1:L, :, :]  # (L H B)

ud = reshape(S4D_jl.D, 1, 10, 1) .* u
y_2 = y_1 + ud

y_3 = S4D_jl.dropout(Flux.gelu(y_2))  # TODO: replace with DropoutND
y_4 = glu(conv(y_3), 2)  # Some differences here with glu
y_4 = S4D_jl.out(y_3)

maximum(permutedims(y_4, (3,2,1)) .- py"y_4.detach().numpy()")  # 5.1870942f-5
