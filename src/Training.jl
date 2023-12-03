include("AutoEncoders.jl")
include("Connections.jl")
include("Losses.jl")

using Printf, Interpolations, CUDA, Serialization, FileIO, HTTP, HTTP.WebSockets, Flux.Optimise, BSON
using HTTP.WebSockets: isclosed

struct Trainer

    model::AutoEncoder
    optimizer::Flux.Optimise.AbstractOptimiser
    loss::Function

end

function (state::Trainer)(data::AbstractArray)

    ps = Flux.params( state.model )

    gs = Flux.gradient( ps ) do 

        return state.loss( data )

    end

    Flux.update!(state.optimizer, ps, gs)

end

function train( state::Trainer, data )

    trainmode!( state.model )

    for x in data

        state( x )

    end

end

Flux.cpu( state::Trainer ) = Trainer( state.model |> cpu, state.optimizer |> cpu. state.loss )
Flux.gpu( state::Trainer ) = Trainer( state.model |> gpu, state.optimizer |> gpu, state.loss )

function save( save_path::String, model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser )
    serialize(save_path,  Dict("model" => model |> cpu, "optimizer" => optimizer |> cpu))
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()
end