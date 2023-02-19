include("Visualizers.jl")

using Base.Threads

###############
#   Generic   #
###############



function elbo_loss( model::AutoEncoder; true_alpha=fill(0.98, length(model.interpret.bias)), burn_in=2^20 )

    true_alpha = convert(model, true_alpha)

    mme = alpha_mme(true_alpha)

    return function (decoder::AbstractArray, latent::AbstractArray, alpha::AbstractArray)
        
        mme_alpha, N = @ignore mme(latent)

        true_alpha   = @ignore N > log2(burn_in) ? convert(model, mme_alpha) : true_alpha

        return ELBO(decoder, alpha, true_alpha)

    end

end



function visualize_loss( model::AutoEncoder )

    visualize = visualizer( model )

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

    elbo, reconstruction, visualizer, printer = fetch.((

        (@spawn elbo_loss( model )),
        (@spawn reconstruction_loss( model )),
        (@spawn visualize_loss( model )),
        (@spawn print_loss("\nr_loss %.5e -elbo %.5e"))

    ))

    return function ( x::AbstractArray )

        e, y, α, β, l = model(x)

        E = elbo( e, l, α )

        R = reconstruction(x, y)

        @ignore visualizer(x, y)

        @ignore printer(R, -E)

        return R - E * 1f-2

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