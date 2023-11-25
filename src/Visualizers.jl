include("DataIterators.jl")
include("ResNet.jl")

using JSON

#################
#   ResNetVAE   #
#################

function process( x::AbstractArray )

    x = x .|> N0f8

    return reinterpret( UInt8, reshape(x, reduce(*, size(x))) )

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

        ) |> JSON.json

        put!( c, metadata |> Vector{UInt8} )

    end

end