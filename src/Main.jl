using Pkg

Pkg.add( ["BSON", "Colors", "CUDA", "Distributions", "FFTW", "FileIO", "Flux", "HTTP", "Images", "ImageTransformations", "Interpolations", "JSON", "LibSndFile", "LinearAlgebra", "NNlib", "Plots", "Printf", "Random", "Serialization", "SliceMap", "SpecialFunctions", "Statistics", "VideoIO", "WAV", "Zygote"] )

include("Training.jl")
include("Frontend.jl")

const prematch_env( x::String ) = [ last(y) for y in [ENV...] if occursin(x, first(y)) ]

if "frontend-server" in ARGS

    HTTP.serve( ENV["FRONTEND_HOST"], parse(Int, ENV["FRONTEND_PORT"]), verbose=true ) do request::HTTP.Request

        return router( request )

    end

end



if "data-server" in ARGS

    DataServer( host=ENV["DATA_HOST"], port=parse(Int, ENV["DATA_PORT"]) )

end



if "training-server" in ARGS

    model = ResNetVAE( 64 )

    convert( Float32, model )

    opt  = Optimiser( ClipNorm(1f0), ADAM(1f-3), NoNaN() )

    vh = prematch_env("VISUALIZER_HOST")
    vp = parse.(Int, prematch_env("VISUALIZER_PORT")) 

    loss = create_loss_function( model, visualizer_hosts=vh, visualizer_ports=vp )

    trainer = Trainer( model, opt, loss )

    for image in Connection( host=ENV["DATA_HOST"], port=parse(Int, ENV["DATA_PORT"]) )

        trainer( image .|> Float32 )

    end
    
end

