include("DataIterators.jl")
include("AutoEncoders.jl")
include("ResNet.jl")

#################
#   ResNetVAE   #
#################

function from_color( image::AbstractArray )

    return RGB.( image[:, :, 1], image[:, :, 2], image[:, :, 3] )

end

function visualizer( model::ResNetVAE, kwargs... )

    args = Dict(kwargs...)

    cx = Channel(1)
    cy = Channel(1)

    X = @async DataServer( host=args[:visualizer_hosts][1], port=args[:visualizer_ports][1], iterator=cx, save=false )
    Y = @async DataServer( host=args[:visualizer_hosts][2], port=args[:visualizer_ports][2], iterator=cy, save=false )

    return function( x::AbstractArray, y::AbstractArray )

        put!( cx, x[:, :, :, 1] .|> N0f8 )
        put!( cy, y[:, :, :, 1] .|> N0f8 )

    end

end