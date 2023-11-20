include("DataIterators.jl")
include("ResNet.jl")

#################
#   ResNetVAE   #
#################

function from_color( image::AbstractArray )

    return RGB.( image[:, :, 1], image[:, :, 2], image[:, :, 3] )

end

function visualizer( model::ResNetVAE )

    cx = Channel{Vector{UInt8}}()
    cy = Channel{Vector{UInt8}}()

    X = @async WSServer( cx, port=parse(Int, ENV["VISUALIZER_PORT_1"]) )
    Y = @async WSServer( cy, port=parse(Int, ENV["VISUALIZER_PORT_2"]) )

    function process( x::AbstractArray )

        sz = Vector{UInt8}( "$(eltype(x));$(reduce(*, "$s " for s in size(x)))\n" )
    
        return [ sz..., reinterpret(UInt8, x[:, :, :, 1])... ]

    end

    return function( x::AbstractArray, y::AbstractArray )

        x = process(x)
        y = process(y)
        
        put!( cx, x )
        put!( cy, y )

    end

end