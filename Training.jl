include("ResNetREPLVisualizers.jl")


using Printf, Interpolations, BSON, ImageView, Gtk, CUDA
using Flux: @epochs


function save( save_path::String, model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser )

    BSON.bson(save_path,  Dict("model" => model |> cpu, "optimizer" => optimizer |> cpu))
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()

    println("model saved!")

end



function train( model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser, loss_fn::Function, data::Union{DataIterator, BatchIterator}, save_path::String; save_freq=10, epochs=1 )

    trainmode!(model)

    parameters = Flux.params(model)

    for _ in 1:epochs

        callback = Flux.throttle( () -> save( save_path, model, optimizer ), save_freq )

        Flux.Optimise.train!( loss_fn, parameters, data, optimizer, cb=callback )

    end

end
