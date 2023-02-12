include("Training.jl")

using ImageView, ArgParse, Flux.Optimise, Gtk, BSON, .ResNet, .DDSP, .AutoEncoders

function arguments()

    s = ArgParseSettings()

    @add_arg_table s begin 

        "--train"
            action = :store_true

        "--shuffle"
            action = :store_true

        "--shuffle-dir"
            action = :store_true

        "--no-gpu"
            action = :store_true

        "--type"
            arg_type = String
            default  = "image"

        "--load"
            arg_type = String

        "--learning-rate"
            arg_type = String
            default  = "1e-3"

        "--iterations"
            arg_type = Int
            default  = 1000

        "--batches"
            arg_type = Int
            default  = 1

        "--epochs"
            arg_type = Int
            default  = 1

        "--save-freq"
            arg_type = Int
            default  = 10

        "--model-size"
            arg_type = Int
            default  = 64

        "--audio-size"
            arg_type = Int
            default  = 2^20

    end

    return parse_args( s )

end

args = arguments()

model_size = args["model-size"]

device     = args["no-gpu"] ? cpu : gpu

lr         = parse(Float64, args["learning-rate"])

data_args  = filter( kv -> !isnothing( last(kv)), Dict(

    :directory   => "data/$(args["type"])",
    :batches     => args["batches"],
    :shuffle     => args["shuffle"],
    :shuffle_dir => args["type"] != "image" ? args["shuffle-dir"] : nothing,
    :sample_size => args["type"] == "audio" ? args["sample_size"] : nothing

))

models = Dict(

    "image" => () -> ( 
        
        model      = ResNetVAE(model_size, device=device),
        optimizer  = Optimiser( ClipNorm(1f0), NoNaN(), ADAM(lr, (0.9, 0.99)) )


    ), 

    "audio" => () -> ( 
        
        model      = AudioVAE(model_size, device=device),
        optimizer  = Optimiser( ClipNorm(1f0), NoNaN(), ADAM(lr, (0.9, 0.99)) )
        
    )

)

losses = Dict(

    Main.ResNet.AutoEncoders.AutoEncoder => ResNet.loss,
    Main.DDSP.AutoEncoders.AutoEncoder   => DDSP.loss

)

data_iterators = Dict(

    Main.ResNet.AutoEncoders.AutoEncoder => () ->  ImageIterator(;data_args...),
    Main.DDSP.AutoEncoders.AutoEncoder   => () ->  AudioIterator(;data_args...)

)

if isnothing(args["load"])

    filename = "data/models/$(args["type"])$model_size.bson" 
    model, optimizer = models[args["type"]]() |> values    

else

    filename = "data/models/$(args["load"]).bson"
    loaded   = BSON.load(filename)

    model, optimizer = loaded["model"], loaded["optimizer"]

end

output_name = args["type"] == "audio" ? "audio_test.wav" :
              args["type"] == "image" ? "image_test.jpg" :
              args["type"] == "video" ? "video_test.mp4" :
              "test"


save_func   = args["type"] == "audio" ? save_audio :
              args["type"] == "image" ? save_video :
              args["type"] == "video" ? save_video :
              exit()


model = model |> device

if args["train"]

    loss = model |> losses[typeof(model)]

    data = (data_iterators[typeof(model)])()

    train_autoencoder( model, optimizer, loss, data, filename, save_freq=args["save-freq"], epochs=args["epochs"] )

end

