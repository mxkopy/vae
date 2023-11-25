include("DataIterators.jl")
include("ResNet.jl")

using JSON

#################
#   ResNetVAE   #
#################

function process( x::AbstractArray )

    x = x .|> N0f8

    metadata = JSON.json( [] )

    metadata = Vector{UInt8}( "$(eltype(x));$(reduce(*, "$s " for s in size(x)))\n" )

    payload  = reinterpret( UInt8, reshape(x, reduce(*, size(x))) )

    return vcat( metadata, payload )

end


function visualizer( model::ResNetVAE )

    cx = Channel{Vector{UInt8}}()
    cy = Channel{Vector{UInt8}}()

    X = @async WSServer( cx, port=parse(Int, ENV["VISUALIZER_PORT_1"]) )
    Y = @async WSServer( cy, port=parse(Int, ENV["VISUALIZER_PORT_2"]) )

    return function( x::AbstractArray, y::AbstractArray )

        metadata = JSON.json( [ [size(x)...], [size(y)...] ] ) * '\n';

        put!( cx, x |> process )
        put!( cy, y |> process )

    end

end