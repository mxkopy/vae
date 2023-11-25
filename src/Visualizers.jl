include("DataIterators.jl")
include("ResNet.jl")

using JSON

#################
#   ResNetVAE   #
#################

function process( x::AbstractArray )

    h, w = size(x, 1), size(x, 2)

    x = x .|> N0f8
    x = reinterpret(UInt8, x)

    y = zeros(UInt8, length(x) + length(x) ÷ 3)

    i = 0
    k = 0

    # there has to be a more elegant way

    while i < h * w

        y[0 + i * 4 + 1] = x[k + 0 * h * w + 1]
        y[1 + i * 4 + 1] = x[k + 1 * h * w + 1]
        y[2 + i * 4 + 1] = x[k + 2 * h * w + 1]
        y[3 + i * 4 + 1] = 255

        i += 1
        k += 1

    end

    return y

end



function visualizer( model::ResNetVAE )

    channel = Channel{Vector{UInt8}}()

    @async WSServer( channel, port=parse(Int, ENV["TRAINING_PORT"]) )

    return function( x::AbstractArray, y::AbstractArray )

        input  = process(x)
        output = process(y)

        metadata = Dict(

            "input"  => Dict(
                "height" => size(x, 1),
                "width"  => size(x, 2),
                "size"   => sizeof(input)
            ),

            "output" => Dict(
                "height" => size(y, 1),
                "width"  => size(y, 2),
                "size"   => sizeof(output)
            )

        ) |> JSON.json |> Vector{UInt8} 

        message = vcat( metadata, [0], input, output )

        put!( channel, message )

    end

end