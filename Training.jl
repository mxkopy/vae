include("AutoEncoderOutputs.jl")


using Printf, Interpolations, BSON, Zygote, ImageView, Gtk, CUDA
using Flux: @epochs


function save( filename::String, model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser )

    device = deepcopy(model.device)

    BSON.bson(filename,  Dict("model" => model |> cpu, "optimizer" => optimizer |> cpu))
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()

    println("model saved!")

    model, optimizer = model |> device, optimizer |> device

end



function train( model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser, loss::Function, data::DataIterator, filename::String; save_freq=10, epochs=1 )

    trainmode!(model)

    parameters = Flux.params(model)

    for _ in 1:epochs

        callback = Flux.throttle( () -> save( filename, model, optimizer ), save_freq )

        Flux.Optimise.train!( loss, parameters, data, optimizer, cb=callback )

    end

end
