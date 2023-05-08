using PyCall

py"""
import sys
sys.path.insert(0, "/Users/gregszumel/Documents/coding/s4")
"""

s4d_py = pyimport("s4d")

get_param_vals(parameter_tensor)  = parameter_tensor.data.numpy()

function get_s4d_kernel_params(k)
    return (get_param_vals(k.log_A_real), get_param_vals(k.A_imag),
            get_param_vals(k.C), get_param_vals(k.log_dt))
end

function test_s4d_jl_implementation()
    s4dk = s4d_py.S4DKernel(10)  # builds kernel
    # builds equivalent julia kernel
    log_A_real, A_imag, C, log_dt = get_s4d_kernel_params(s4dk)
    s4dk_jl = S4DKernel(log_A_real, A_imag, C[:, :, 1], C[:, :, 2], log_dt)
    jl = s4dk_jl(10, 20)  # runs julia kernel 
    np = s4dk(20).detach().numpy()  # runs np kernel
    @assert maximum(jl - np) < 0.001
end

function get_s4d_params(m)
    return Dict(name => get_param_vals(param) 
                for (name, param) in m.named_parameters())
end


