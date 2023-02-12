include("Visualizers.jl")


#################
#   ResNetVAE   #
#################



function interpolate_data(out::AbstractArray, data::AbstractArray)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), size(out) )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), size(out) ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end

# this creates a loss function that's used in training
# includes stuff like visualizers & scheduling
# i don't imagine super abstract usage so for now many things are hardcoded

# schedule_period is a pair giving the number of iterations for the annealing & fixed loss, respectively
# from https://arxiv.org/pdf/1903.10145.pdf

function loss( model::ResNetVAE; α_true=fill(0.98, length(model.alpha.bias)) .|> model.precision |> model.device, burn_in=10000, schedule_period=400=>250 )

    visualize = visualizer(model)

    schedule  = vcat(range(0.0, 1f-3, first(schedule_period)), fill(1.0, last(schedule_period))) |> Iterators.cycle |> Iterators.Stateful

    mme, n    = alpha_mme(α_true), 0

    return @noinline function(data::AbstractArray)

        encoder, decoder, α, β, latent = model(data)

        data    = @ignore interpolate_data(decoder, data) .|> model.precision |> model.device
        
        w       = @ignore first(schedule)

        α_t     = @ignore mme(latent)

        α_true  = @ignore (n += 1) > burn_in ? α_t : α_true

        elbo    = ELBO(decoder, α, α_true)

        r_loss  = Flux.Losses.mse(decoder, data)

        Zygote.ignore() do  
    
            visualize(data, decoder)
    
            @printf "\nr_loss %.5e -elbo %.5e %i" r_loss -elbo n
            flush(stdout)
    
        end

        return r_loss - elbo * 1f-2

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

function loss(model::DDSP)

    return nothing

end


