"""
    DropoutNd - tied dropout across the specified dimensions. Takes a percentage
    of dropout and a tie function. 
"""
struct DropoutNd
    p::Float32
    tie::Bool
end

# simplifying generation
DropoutNd(x::Float32) = DropoutNd(x, true)
DropoutNd() = DropoutNd(.5f0, true)
# case where inactive, just return X; otherwise run through the dropout
(d::DropoutNd)(X::Array, active::Bool) = active && d.p != 0 ? d(X) : X

"""
Input: Array of dimension [L, H, B]
"""
function (d::DropoutNd)(X::Array)
    mask_shape = size(X)[1:2] 
    mask = rand(Float32, mask_shape...) .< (1.0f0 - d.p)
    return X .* mask .* (1.0f0/(1.0f0-d.p))
end
