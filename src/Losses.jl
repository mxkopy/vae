include("Visualizers.jl")

###############
#   Generic   #
###############



function elbo_loss( model::AutoEncoder )

    return function( z_0::AbstractArray )

        s = 0

        for c in Base.Iterators.product( axes(z_0)[2:end]... )

            s += FEB( model.flow, z_0[:, c...] )

        end

        return s

    end

end



function visualize_loss( model::AutoEncoder, kwargs... )

    visualize = visualizer( model, kwargs... )

    return function ( data... )

        @async visualize( data... )

    end

end



function print_loss( format::String="\nr_loss %.5e -elbo %.5e" )

    @eval return function( losses... )

        @printf $format losses...

        flush(stdout)

    end

end



function create_loss_function( model::AutoEncoder )

    E, R, V, P = (

        elbo_loss( model ),
        reconstruction_loss( model ),
        visualize_loss( model ),
        print_loss("\nr_loss %.5e -elbo %.5e")

    )

    return function ( x::AbstractArray )

        e, μ, σ, z_0, y = model(x)

        e = E(z_0)

        r = R(x, y)

        @ignore V(x, y)

        @ignore P(r, -e)

        return r + e

    end

end


#################
#   ResNetVAE   #
#################

function interpolate_data(data::AbstractArray, out_size::Tuple)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), out_size )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), out_size ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end

function reconstruction_loss( model::ResNetVAE )

    return function ( x::AbstractArray, y::AbstractArray )

        x = @ignore convert(model, interpolate_data(x, y |> size))

        return Flux.Losses.mse( x, y )

    end

end
