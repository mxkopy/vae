include("Visualizers.jl")

struct ReconstructionLoss{T <: AutoEncoder}
    model::T
end

function (r::ReconstructionLoss{ResNetVAE})(x::AbstractArray, y::T) where T <: AbstractArray
    x = interpolate_data(x, size(y)) |> T
    return Flux.Losses.mse( x, y )
end

struct ELBOLoss{T <: AutoEncoder}
    model::T
end

function (elbo::ELBOLoss{ResNetVAE})(z_0::AbstractArray{P}) where P
    s::P = 0
    for c in Base.Iterators.product( axes(z_0)[2:end]... )
        s += FEB( elbo.model.flow.layer, z_0[:, c...] )
    end
    return s
end

struct Printer{T <: AutoEncoder}
    model::T
    format::Function
end

Printer(model::ResNetVAE) = Printer(model, (r_loss::Number, e_loss::Number) -> "\nr_loss $(string(r_loss)) -elbo $(string(e_loss))" )

@eval function (printer::Printer)( losses... )
    printer.format(losses...) |> print
    flush(stdout)
end


function create_loss_function( model::ResNetVAE )

    E, R, V, P = ELBOLoss(model), ReconstructionLoss(model), Visualizer(model), Printer(model)

    return function ( x::AbstractArray )

        e, μ, σ, z_0, f, y = model(x)

        e = E(z_0)

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
