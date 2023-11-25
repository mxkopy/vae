include("DataIterators.jl")
include("ResNet.jl")

using JSON

#################
#   ResNetVAE   #
#################

function process( x::AbstractArray )

    h, w = size(x, 1), size(x, 2)

    x = x .|> N0f8

    y = zeros(UInt8, length(x) + length(x) / 3)

    i = 0
    k = 0

    # there has to be a more elegant way

    while i < h * w

        y[0 + i * 4 + 1] = x[k + 0 * h * w + 1]
        y[1 + i * 4 + 1] = x[k + 1 * h * w + 1]
        y[2 + i * 4 + 1] = x[k + 2 * h * w + 1]
        y[3 + i * 4 + 1] = 255

        i++
        k++

    end

    return y

end


function visualizer( model::ResNetVAE )

    channel = Channel{Vector{UInt8}}()

    @async WSServer( channel, port=parse(Int, ENV["TRAINING_PORT"]) )

    return function( input::AbstractArray, output::AbstractArray )

        metadata = Dict(

            "input"  => (
                "size" => size(input)
            ),

            "output" => (
                "size" => size(output)
            )

        ) |> JSON.json |> Vector{UInt8} 

        message = vcat(metadata, [0], process(input), process(output))

        put!( channel, message )

    end

end