# include("deps.jl")
include("Training.jl")
include("Frontend.jl")

const prematch_env( x::String ) = [ last(y) for y in [ENV...] if occursin(x, first(y)) ]

if "frontend-server" in ARGS

    HTTP.serve( "0.0.0.0", parse(Int, ENV["FRONTEND_PORT"]), verbose=true ) do request::HTTP.Request

        return router( request )

    end

end



if "data-server" in ARGS

    iterator = BatchIterator( ImageReader(ENV["DATA_TARGET"]), parse(Int, ENV["BATCHES"]) )

    DataServer( host="0.0.0.0", port=parse(Int, ENV["DATA_PORT"]), iterator=iterator )

end



if "training-server" in ARGS

    model = ResNetVAE( 64 )

    convert( Float32, model )

    opt  = Optimiser( ClipNorm(1f0), ADAM(1f-3), NoNaN() )

    vh = [ "0.0.0.0", "0.0.0.0" ]
    vp = parse.(Int, prematch_env("VISUALIZER_PORT")) 

    loss = create_loss_function( model, visualizer_hosts=vh, visualizer_ports=vp )

    trainer = Trainer( model, opt, loss )

    for image in Connection( host="ws://data", port=parse(Int, ENV["DATA_PORT"]) )

        trainer( image .|> Float32 )

    end
    
end

