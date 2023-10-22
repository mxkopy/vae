include("Flows.jl")

using Flux, CUDA, Serialization, Distributions, SpecialFunctions, Printf

using Zygote: @ignore

import Flux.outputsize
import Flux.trainmode!

abstract type AutoEncoder end


function sample_gaussian( μ::T, σ::T ) where T <: Number

    ϵ = @ignore rand( Normal(T(0), T(1)) )

    # ϵ = @ignore 1

    return μ + σ * ϵ

end


macro autoencoder( T::Symbol )
        
    return eval(:( 
        
        struct $T <: AutoEncoder

            encoder
            decoder

            μ
            σ

            flow::Flow

        end;
        
        Flux.@functor $T (encoder, decoder, μ, σ, flow);

    ))

end

function (model::AutoEncoder)(data::AbstractArray)

    # data       = @ignore convert(model, data)

    enc_out    = model.encoder( data )

    μ          = model.μ(enc_out)
    σ          = model.σ(enc_out)

    z_0        = sample_gaussian.( μ, σ )

    flow       = model.flow( z_0 )

    dec_out    = model.decoder( flow )

    return (enc_out=enc_out, μ=μ, σ=σ, z_0=z_0, dec_out=dec_out)

end


Flux.gpu(x::AutoEncoder) = fmap(gpu, x)
Flux.cpu(x::AutoEncoder) = fmap(cpu, x)

convert(T::DataType, model::AutoEncoder)                   = fmap( x -> x isa AbstractArray ? T.(x) : x, model )
convert(model::AutoEncoder, x::AbstractArray)              = x .|> eltype( Flux.params(model) |> first )
convert(model::Union{AutoEncoder, Flux.Optimise.AbstractOptimiser}; precision=Float32, device=gpu) = convert(precision, model) |> device


struct NoNaN <: Flux.Optimise.AbstractOptimiser end

function Flux.Optimise.apply!(o::NoNaN, x, Δ::AbstractArray{T}) where T

    sanitize(δ)::T = isnan(δ) || isinf(δ) ? T(0) : δ

    Δ = sanitize.(Δ)

end
