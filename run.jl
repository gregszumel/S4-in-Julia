using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Flux: onehotbatch, onecold, crossentropy, Momentum
using Base.Iterators: partition
import MLUtils

include("models.jl")  # S4Model

# setting up parameters
learning_rate = 0.01
weight_decay = 0.01

epochs = 100
dataset = :cifar10
grayscale = true

num_workers = 4
batch_size=32
n_layers=4
d_model=128
dropout=0.1
prenorm=true

device = "cpu";
best_acc = 0; start_epoch = 0;

model = S4Model(1024, 10, d_model)  # d_model = 1024, d_state = 64

train = CIFAR10(split=:train)[:]
test = CIFAR10(split=:test)[:]
train, val = MLUtils.splitobs((train.features, train.targets), at=0.9)

train_dataloader = Flux.DataLoader(train, batchsize=batch_size)
val_dataloader = Flux.DataLoader(val, batchsize=batch_size)
test_dataloader = Flux.DataLoader(test, batchsize=batch_size)

loss(x, y) = sum(crossentropy(x, y))
accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))
opt_state = Flux.setup(Adam(), model)

epochs = 10

for (input, label) in train_dataloader
    print(input |> size, label |>size)
    global output = model(reshape(input, (3, 1024, batch_size)))
    global i = input
    global l = label
    break
end
output
loss(Flux.sigmoid.(output), onehotbatch(l, 0:9))

for (input, label) in train_dataloader
    print(input |> size, label |>size)
    gs = Flux.gradient(model) do m
        output = m(reshape(input, (3, 1024, batch_size)))
        l = loss(Flux.sigmoid.(output), onehotbatch(label, 0:9))
        println(l)
        l
    end
    Flux.update!(opt_state, model, gs[1])
    break
end
@show accuracy(valX, valY)


for epoch = 1:epochs
    for d in zip(train_x, labels)
      inputs, labels = d
      gs = Flux.gradient(model) do m
        l = loss(m(inputs), labels)
        println(l)
        l
      end
      Flux.update!(opt_state, model, gs[1])
    end
    @show accuracy(valX, valY)
  end
  

