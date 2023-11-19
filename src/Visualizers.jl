include("DataIterators.jl")
include("ResNet.jl")

#################
#   ResNetVAE   #
#################

function from_color( image::AbstractArray )

    return RGB.( image[:, :, 1], image[:, :, 2], image[:, :, 3] )

end

function visualizer( model::ResNetVAE )

    cx = Channel(1)
    cy = Channel(1)

    X = @async WSServer( host="0.0.0.0", port=parse(Int, ENV["VISUALIZER_PORT_1"]), iterator=cx, save=false )
    Y = @async WSServer( host="0.0.0.0", port=parse(Int, ENV["VISUALIZER_PORT_2"]), iterator=cy, save=false )

    return function( x::AbstractArray, y::AbstractArray )

        put!( cx, x[:, :, :, 1] .|> N0f8 )
        put!( cy, y[:, :, :, 1] .|> N0f8 )

    end

end