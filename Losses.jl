include("Visualizers.jl")

###############
#   Generic   #
###############



function elbo_loss( model::AutoEncoder; true_alpha=fill(model.precision(0.98), length(model.alpha.bias)) |> model.device, burn_in=1e10 )

    mme = alpha_mme(true_alpha)

    return function (decoder::AbstractArray, latent::AbstractArray, alpha::AbstractArray)
        
        mme_alpha, N = @ignore mme(latent)

        true_alpha   = @ignore N > log(burn_in) ? mme_alpha : true_alpha

        return ELBO(decoder, alpha, true_alpha)

    end

end



function visualize_loss( model::AutoEncoder )

    visualize = visualizer( model )

    return function ( data... )

        visualize( data... )

    end

end



function print_loss( format )

    return function( losses... )

        @eval @printf $format $(losses...)

        flush(stdout)

    end

end


function create_loss_function( model::AutoEncoder )

    elbo           = elbo_loss( model )

    reconstruction = reconstruction_loss( model ) 

    visualizer     = visualize_loss( model )

    printer        = print_loss("\nr_loss %.5e -elbo %.5e")

    return function ( x::AbstractArray )

        e, y, α, β, l = model(x)

        E = elbo( e, l, α )

        R = reconstruction(y, x)

        @ignore @async visualizer(y, x)

        @ignore @async printer(R, -E)

        return R - E * 1f-2

    end

end


#################
#   ResNetVAE   #
#################

function interpolate_data(out::AbstractArray, data::AbstractArray)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), size(out) )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), size(out) ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end

function reconstruction_loss( model::ResNetVAE )

    return function ( y::AbstractArray, x::AbstractArray )

        return Flux.Losses.mse( y, @ignore interpolate_data(y, x) .|> model.precision |> model.device )

    end

end


############
#   DDSP   #
############



function spectral_distance( out::AbstractArray, data::AbstractArray; ϵ=1f-8 )

    F  = a -> abs.(a) .^ 2 |> sum |> sqrt
    L1 = a -> abs.(a)      |> sum 

    Y = fft(out,  [1])
    X = fft(data, [1])

    A = F(X .- Y) / max( F(X), ϵ )
    B = log( L1( X .- Y ) )

    return A + B

end

function reconstruction_loss( model::DDSP )

    return function ( y::AbstractArray, x::AbstractArray )

        return Flux.Losses.mse( y, x ) + spectral_distance(y, x)

    end

end