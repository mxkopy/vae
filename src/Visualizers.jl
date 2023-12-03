include("DataIterators.jl")
include("ResNet.jl")

using JSON

struct Visualizer{Model}
    model::Model
    channel::Channel{Vector{UInt8}}
end

function Visualizer(model::AutoEncoder; port=parse(Int, ENV["TRAINING_PORT"]))
    channel = Channel{Vector{UInt8}}()
    @async WSServer(channel, port=port)
    return Visualizer(model, channel)
end

function (visualizer::Visualizer{ResNetVAE})(x::AbstractArray, y::AbstractArray)

    x = x |> cpu
    y = y |> cpu
    
    x_info = add_info(x, height=size(x, 1), width=size(x, 2), name="input")
    y_info = add_info(y, height=size(y, 1), width=size(y, 2), name="output")

    x_bits = process_raw_image(x)
    y_bits = process_raw_image(y)

    images = [(info=x_info, bits=x_bits), (info=y_info, bits=y_bits)]

    message  = to_message( images )

    # put!( visualizer.channel, message )

end
