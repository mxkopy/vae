include("Training.jl")
include("DataIterators.jl")
include("Frontend.jl")

if "frontend" in ARGS


    HTTP.serve( "0.0.0.0", parse(Int, ENV["FRONTEND_PORT"]), verbose=true ) do request::HTTP.Request

        return router( request )

    end

end



if "data" in ARGS


    iterator = BatchIterator( ImageReader(ENV["DATA_TARGET"]), parse(Int, ENV["BATCHES"]) )

    DataServer( host="0.0.0.0", port=parse(Int, ENV["DATA_PORT"]), iterator=iterator )

end



if "training" in ARGS

    model = ResNetVAE( 64 )

    convert( Float32, model )

    opt  = Optimiser( ClipNorm(1f0), ADAM(1f-3), NoNaN() )

    loss = create_loss_function( model )

    trainer = Trainer( model, opt, loss ) |> gpu

    for image in DataClient( host="ws://" * ENV["DATA_HOST"], port=parse(Int, ENV["DATA_PORT"]) )

        trainer( image .|> Float32 .|> gpu )

    end
    
end

