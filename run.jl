using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Flux: onehotbatch, onecold, crossentropy, Momentum
using Base.Iterators: partition
import MLUtils

include("julia/models.jl")  # S4Model

# setting up parameters
learning_rate = 0.1
weight_decay = 0.1

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

model = S4Model(3, 10, d_model, n_layers, 0.1f0, true)  # d_model = 1024, d_state = 64
copied_model = deepcopy(model)

train = CIFAR10(split=:train)[:]
test = CIFAR10(split=:test)[:]
train, val = MLUtils.splitobs((train.features, train.targets), at=0.9)

train_dataloader = Flux.DataLoader(train, batchsize=batch_size, partial=false)
val_dataloader = Flux.DataLoader(val, batchsize=batch_size, partial=false)
test_dataloader = Flux.DataLoader(test, batchsize=batch_size, partial=false)


loss(x, y) = sum(crossentropy(x, y))
accuracy(x, y) = mean(onecold(model(x), 0:9) .== y)
opt_state = Flux.setup(Adam(), model)

epochs = 10

function test_forward()
    for (input, label) in train_dataloader
        print(input |> size, label |>size)
        global output = model(reshape(input, (1024, 3, batch_size)))
        global i = input
        global l = label
        break
    end
    return true
end

function test_forward_backward()
    for (input, label) in train_dataloader
        gs = Flux.gradient(model) do m
            output = m(reshape(input, (1024, 3, batch_size)))
            l = loss(Flux.softmax(output), onehotbatch(label, 0:9))
            println(l)
            global outputs = output
            l
        end
        Flux.update!(opt_state, model, gs[1])
        return input, label, outputs
    end
end

function test_convergence()
    for _ in 1:100
        test_forward_backward()
    end
end


for _ in 1:epochs
    for (input, label) in train_dataloader
        gs = Flux.gradient(model) do m
            output = Flux.softmax(m(reshape(input, (1024, 3, batch_size))))
            l = loss(output, onehotbatch(label, 0:9))
            l
        end
        Flux.update!(opt_state, model, gs[1])
        break
    end
    correct = 0
    total = 0
    for (input, label) in val_dataloader
        output = model(reshape(input, (1024, 3, batch_size)))
        correct += sum(onecold(Flux.softmax(output)) .- 1 .== label)
        total += 32
        break
    end
    println(correct / total)
end

model.enc.dense.weight
copied_model.enc.dense.weight

model.s4blocks[3]
copied_model.s4blocks[3]

model.dec.weight
copied_model.dec.weight

println(model)
println(copied_model)