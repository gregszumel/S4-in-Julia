using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Flux: onehotbatch, onecold, crossentropy, Momentum
using Base.Iterators: partition
import MLUtils

include("models.jl")  # S4Model

train_x, train_y = CIFAR10(split=:train)[:]
labels = onehotbatch(train_y, 0:9)

loss(x, y) = sum(crossentropy(x, y))
opt = Momentum(0.01)
model = S4Model(1024)  # d_model = 1024, d_state = 64
opt_state = Flux.setup(Adam(), model)

example = train_x[:, :, :, 1:10]
model(reshape(example, (3, 1024, 10)))

# accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))

# epochs = 10

# for epoch = 1:epochs
#     for d in train
#       inputs, labels = d
#       gs = Flux.gradient(m2) do m
#         l = loss(m(inputs), labels)
#         println(l)
#         l
#       end
#       Flux.update!(opt_state, m2, gs[1])
#     end
#     @show accuracy(valX, valY)
#   end
  

