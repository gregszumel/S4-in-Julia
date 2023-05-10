"""
    Gated Linear Unit activation function. Used in S4.
    Input: 
        A - a multidim-array with an even ~dim~ 
        dim - the dim to divide by.
    Output:
        B .* σ.(C), where B and C are the respective first and second halves 
        of A split on dimension dim 
"""
function glu(A::Array, dim=-1)
    if dim < 0
        dim = ndims(A) + dim + 1
    end
    @assert (size(A, dim) % 2) == 0 string("input not even, size ", size(A)," on ", dim)
    halfway_thru_dim = Int(size(A)[dim]/2)
    a = selectdim(A, dim, 1:halfway_thru_dim)
    b = selectdim(A, dim, halfway_thru_dim+1:size(A)[dim])
    return a .* Flux.sigmoid(b)
end
