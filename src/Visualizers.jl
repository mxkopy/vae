include("DataIterators.jl")
include("ResNet.jl")

using JSON

function process( x::AbstractArray )

    x = permutedims(x, (3, 2, 1, 4))

    x = colorview(RGB, x) .|> RGBA{N0f8}

    x = channelview(x)

    x = reinterpret(UInt8, x)

    # 3 2 1 x
    # 2 3 1 x
    # 3 1 2 x
    # 1 3 2 ?
    # 1 2 3 - 
    # 2 1 3

    x = permutedims(x, (1, 3, 2, 4))

    x = reshape(x, length(x))

    x = Vector{UInt8}(x)

    return x

end



function visualizer( model::ResNetVAE )

    channel = Channel{Vector{UInt8}}()

    @async WSServer( channel, port=parse(Int, ENV["TRAINING_PORT"]) )

    return function( x::AbstractArray, y::AbstractArray )

        x_info = add_info(x, height=size(x, 1), width=size(x, 2), name="input")
        y_info = add_info(y, height=size(y, 1), width=size(y, 2), name="output")

        x_bits = process(x)
        y_bits = process(y)

        images = [(bits=x_bits, info=x_info), (bits=y_bits, info=y_info)]

        message  = to_message( images )

        put!( channel, message )

    end

end