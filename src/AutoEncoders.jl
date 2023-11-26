include("Flows.jl")

using Flux, CUDA, Serialization, Distributions, SpecialFunctions, Printf

using Zygote: @ignore

import Flux.outputsize
import Flux.trainmode!

abstract type AutoEncoder {
    Precision <: Number,
    Function
} end

macro autoencoder( T::Symbol )
        
    return eval(:(
        
        struct $T{Precision, Device} <: AutoEncoder{Precision, Device}

            encoder
            decoder

            m
            s

            flow::Flow

            $T{Precision, Device}( encoder, decoder, m, s, flow ) where {Precision, Device}
            = new{Precision, Device}( encoder, decoder, m, s, flow )

        end;

        Flux.@functor $T (encoder, decoder, m, s, flow);

    ))

end

function sample_gaussian( μ::T, σ::T ) where T <: Number

    ϵ = @ignore rand( Normal(T(0), T(1)) )

    return μ + σ * ϵ

end

function (model::AutoEncoder)(data::AbstractArray)

    # data       = @ignore convert(model, data)

    enc_out    = model.encoder( data )

    m          = model.m(enc_out)
    s          = model.s(enc_out)

    z_0        = sample_gaussian.( m, s )

    flow       = model.flow( z_0 )

    dec_out    = model.decoder( flow )

    return (enc_out=enc_out, m=m, s=s, z_0=z_0, dec_out=dec_out)

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
