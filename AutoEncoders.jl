using Flux, Serialization, WAV, Zygote, Distributions, CUDA, SpecialFunctions, Printf

using SpecialFunctions: gamma as Γ, loggamma as logΓ, digamma as ψ
using Zygote: @ignore

import Flux.outputsize

function inv_gamma(x::T, α::T, β::T) where T

    y::T = (1 / β) * (x * α * Γ(α)) ^ (1 / α)

    return isnan(y) ? T(0) : y

end

function sample_dirichlet(x, α, β)

    return inv_gamma.(x, α, β) ./ sum( inv_gamma.(x, α, β), dims=1 )

end


# from the appendix in https://arxiv.org/abs/1901.02739
# modified to use online means 

function alpha_mme(α_true::AbstractVector{T}) where T

    function update(μ, x, n)

        n̂ = ceil(exp(n))

        μ = μ * (n̂ - 1)
        μ = (μ + x) / n̂

        return μ

    end



    μ1, μ2 = α_true .^ 1, α_true .^ 2

    function S(p::AbstractVector{T}, n)

        μ1, μ2 = update.(μ1, p, n), update.(μ2, p .^ 2, n)

        return sum( (μ1 .- μ2) ./ (μ2 .- μ1 .^ 2) ) / length(p)

    end

    P̄ = α_true .* 0

    function mme(P::AbstractVector{T}, n)

        P̄ = update.(P̄, P, n)

        return S(P, n) .* P̄

    end

    N::T = 0.0

    return function( P::AbstractArray{T} )

        P = reshape( P, first(size(P)), : )

        A = map( 1:last(size(P)) ) do i

            n = N + log( 1 + i / exp(N) )

            return mme( P[:, i], n )

        end

        N += log( 1 + last(size(P)) / exp(N) )

        return last(A)

    end

end


function kl_divergence(q::T, p::T) where T <: Number

    kld    = logΓ(p) - logΓ(q) + (q - p) * ψ(q)

    kld::T = isnan(kld) ? 0f0 : kld

    return kld

end


function kl_divergence(Q::AbstractArray{T}, P::AbstractArray{T}) where T <: Number

    KD = kl_divergence.(Q, P)

    return sum( mean(KD, dims=[2, 3]), dims=1 )

end

function mean_log_probability( out::AbstractArray{T} ) where T <: Number

    out_max = @ignore maximum(out)

    return mean( logsoftmax(out_max .- out, dims=[1, 2, 3]), dims=[1, 2, 3] )

end

function ELBO(out::AbstractArray{T}, alpha::AbstractArray{T}, alpha_parameter::AbstractVector{T}) where T <: Number

    elbo = mean_log_probability(out) .- kl_divergence(alpha, alpha_parameter )

    return sum(elbo)

end

abstract type AutoEncoder end

macro autoencoder( T::Symbol )
        
    return eval(:( 
        
        mutable struct $T <: AutoEncoder

            encoder
            decoder
            alpha
            beta
            decode
        
            precision
            device
        
        end;
        
        Flux.@functor $T (encoder, decoder, alpha, beta, decode);

        function $T(encoder, decoder, model_size; precision=Float32, device=gpu)

            alpha  = Dense(model_size, model_size, softplus, init=Flux.identity_init)
            beta   = Dense(model_size, 1,          softplus, init=Flux.glorot_normal(rng_from_array([1.0/model_size])))
        
            decode = Dense(model_size, model_size, init=Flux.identity_init)
        
            return $T(encoder, decoder, alpha, beta, decode, precision, device) |> device
        
        end;

    ))

end


function (model::AutoEncoder)(data::AbstractArray)

    data       = @ignore data .|> model.precision |> model.device

    enc_out    = model.encoder(data)
    enc_out    = permutedims( enc_out, (3, 2, 1, 4) )

    alpha      = model.alpha(enc_out)
    beta       = model.beta(enc_out)

    param      = @ignore rand(Uniform(0, 1), size(alpha)) .|> model.precision |> model.device
    param      = @ignore param ./ sum(param, dims=1)

    latent     = sample_dirichlet(param, alpha, beta)

    interpret  = model.decode(latent)
    interpret  = permutedims( interpret, (3, 2, 1, 4) )

    dec_out    = model.decoder(interpret)

    return (encoder = enc_out, decoder = dec_out, alpha = alpha, beta = beta, latent = latent)

end


Flux.gpu(x::AutoEncoder) = (x.device=gpu; return fmap(gpu, x))
Flux.cpu(x::AutoEncoder) = (x.device=cpu; return fmap(cpu, x))


Flux.outputsize(model::AutoEncoder, inputsize::Tuple) = Flux.outputsize( model.decoder, Flux.outputsize( model.encoder, inputsize ) )



struct NoNaN <: Flux.Optimise.AbstractOptimiser

end


function Flux.Optimise.apply!(o::NoNaN, x, Δ::AbstractArray{T}) where T

    sanitize(δ)::T = isnan(δ) || isinf(δ) ? 0.0 : δ

    Δ = sanitize.(Δ)

end