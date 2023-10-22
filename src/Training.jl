include("ResNetREPLVisualizers.jl")
include("Connections.jl")

using Printf, Interpolations, CUDA, Serialization, FileIO, HTTP, HTTP.WebSockets, Flux.Optimise
using HTTP.WebSockets: isclosed

struct Trainer{Model}

    model::Model
    optimizer::Flux.Optimise.AbstractOptimiser
    loss::Function

end

function (state::Trainer{ResNetVAE})(data::AbstractArray)

    ps = Flux.params( state.model )

    gs = Flux.gradient( ps ) do 

        return state.loss( data )

    end

    Flux.update!(state.optimizer, ps, gs)

end

function train( state::Trainer, data::Connection )

    trainmode!( state.model )

    for x in data

        state( x )

    end

end

Flux.cpu( state::Trainer ) = Trainer( state.optimizer |> cpu, state.model |> cpu, state.data )
Flux.gpu( state::Trainer ) = Trainer( state.optimizer |> gpu, state.model |> gpu, state.data )

function save( save_path::String, model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser )

    serialize(save_path,  Dict("model" => model |> cpu, "optimizer" => optimizer |> cpu))
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()

end