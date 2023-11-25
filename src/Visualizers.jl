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

    c = Channel{Vector{UInt8}}()

    @async WSServer( c, port=parse(Int, ENV["TRAINING_PORT"]) )

    return function( x::AbstractArray, y::AbstractArray )

        metadata = JSON.json( [ [size(x)...], [size(y)...] ] ) * '\n';

        put!( c, metadata )

    end

end