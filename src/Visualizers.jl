include("DataIterators.jl")
include("ResNet.jl")

using JSON

struct Visualizer
    channel::Channel{Vector{UInt8}}
end

function Visualizer(; host="0.0.0.0", port=parse(Int, ENV["VISUALIZER_PORT"]))
    channel = Channel{Vector{UInt8}}()
    @async WSServer(channel, host=host, port=port)
    return Visualizer(channel)
end

function (visualizer::Visualizer)(X::AbstractArray, Y::AbstractArray)

    for i in axes(x, 4)
    
        x, y = reshape(X[:, :, :, i], size(X)[1:3]..., 1), reshape(Y[:, :, :, i], size(Y)[1:3]..., 1)

        x_info = add_info(x, height=size(x, 1), width=size(x, 2), name="input")
        y_info = add_info(y, height=size(y, 1), width=size(y, 2), name="output")

        x_bits = process_raw_image(x)
        y_bits = process_raw_image(y)

        images = [(info=x_info, bits=x_bits), (info=y_info, bits=y_bits)]

        message  = to_message( images )

        put!( visualizer.channel, message )

    end

end
