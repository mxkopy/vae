include("DataIterators.jl")
include("ResNet.jl")

using JSON

#################
#   ResNetVAE   #
#################

function process( x::AbstractArray )

    x = x .|> N0f8

    metadata = Vector{UInt8}( "$(eltype(x));$(reduce(*, "$s " for s in size(x)))\n" )

    payload  = reinterpret( UInt8, reshape(x, reduce(*, size(x))) )

    return vcat( metadata, payload )

end


function visualizer( model::ResNetVAE )

    channel = Channel{Vector{UInt8}}()

    @async WSServer( channel, port=parse(Int, ENV["TRAINING_PORT"]) )

    return function( input::AbstractArray, output::AbstractArray )

        metadata = Dict( 

            "sizes" => ("input" => size(input), "output" => size(output))

        ) |> JSON.json

        put!( c, metadata |> Vector{UInt8} )

    end

end