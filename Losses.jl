include("Visualizers.jl")

function interpolate_data(out::AbstractArray, data::AbstractArray)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), size(out) )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), size(out) ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end


function elbo_loss( model::AutoEncoder; true_alpha=fill(0.98, length(model.alpha.bias)) .|> model.precision |> model.device, burn_in=10000 )

    mme, n  = alpha_mme(true_alpha), 0

    return function (decoder::AbstractArray, latent::AbstractArray, alpha::AbstractArray)
        
        mme_alpha   = @ignore mme(latent)

        true_alpha  = @ignore (n += 1) > burn_in ? alpha_mme : true_alpha

        return ELBO(decoder, alpha, true_alpha)

    end

end


function visualize_loss( model::AutoEncoder, print_string::String )

    visualize = visualizer( model )

    return function ( data::Tuple, losses::Tuple )

        Zygote.ignore() do 

            visualize( data... )

            @eval @printf $print_string $(losses...)

            flush(stdout)

        end

    end

end


function create_loss_function( model::AutoEncoder )

    elbo           = elbo_loss( model )

    reconstruction = reconstruction_loss( model ) 

    visualize      = visualize_loss( model, "\nr_loss %.5e -elbo %.5e" )

    return function ( data::AbstractArray )

        e, d, α, β, l = model(data)

        E = elbo( e, l, α )

        R = reconstruction(d, data)

        visualize( (d, data), (R, -E) )

        return R - E * 1f-2

    end

end


#################
#   ResNetVAE   #
#################



function reconstruction_loss( model::ResNetVAE )

    return function ( decoder_output::AbstractArray, data::AbstractArray )

        return Flux.Losses.mse( decoder_output, @ignore interpolate_data(decoder_output, data) )

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



# function loss( out, data, alpha, alpha_parameter )
    
#     elbo    = ELBO(out, alpha, alpha_parameter)

#     s_loss  = spectral_distance(out, data)

#     r_loss  = Flux.Losses.mse(out, data)

#     Zygote.ignore() do 

#         @printf "\nr_loss %.5e s_loss %.5e -elbo %.5e" r_loss s_loss -elbo
#         flush(stdout)

#     end

#     return r_loss + s_loss #* 1f-2 - elbo * 1f-2

# end

# function loss(model::DDSP)

#     return nothing

# end


