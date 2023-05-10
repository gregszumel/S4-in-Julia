"""
    DropoutNd - tied dropout across the specified dimensions. Takes a percentage
    of dropout and a tie function. 
"""
struct DropoutNd
    p::Float64
    tie::Bool
end

# simplifying generation
DropoutNd(x::Float64) = DropoutNd(x, true)
DropoutNd() = DropoutNd(.5, true)
# case where inactive, just return X; otherwise run through the dropout
(d::DropoutNd)(X::Array, active::Bool) = active && d.p != 0 ? d(X) : X

"""
Input: Array of dimension [L, H, B]
"""
function (d::DropoutNd)(X::Array)
    mask_shape = size(X)[1:2] 
    mask = rand(mask_shape...) .< (1.0 - d.p)
    X = X .* mask .* (1.0/(1.0-d.p))
    return X
end