include("Visualizers.jl")

struct ReconstructionLoss{T <: AutoEncoder}
    model::T
end

function (r::ReconstructionLoss{ResNetVAE})(x::AbstractArray, y::T) where T <: AbstractArray
    x = @ignore interpolate_data(x, size(y)) |> T
    return Flux.Losses.mse( x, y )
end

struct ELBOLoss{T <: AutoEncoder}
    model::T
end

function (elbo::ELBOLoss{ResNetVAE})(FEB::AbstractArray{P}) where P
    return sum(FEB)
end

function create_loss_function( model::ResNetVAE )

    visualizer_channel = Channel()

    @async DataServer(visualizer_channel, port=parse(Int, ENV["TRAINING_PORT"]))

    E, R, P, V = ELBOLoss(model), ReconstructionLoss(model), Printer(), (x, y) -> @async put!(visualizer_channel, (x |> cpu, y |> cpu))

    return function ( x::AbstractArray )

        e, μ, σ, z_0, f, y, FEB = model(x)

        e = E(FEB)

        r = R(x, y)

        @ignore V(x, y)
        @ignore P(r, -e)

        return r + e

    end

end

function interpolate_data(data::AbstractArray, out_size::Tuple)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), out_size )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), out_size ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end
