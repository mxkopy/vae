include("Util.jl")

using Flux, Distributions, LinearAlgebra, CUDA, Serialization, SpecialFunctions, Printf
using Flux: gradient
using Flux.ChainRulesCore: @ignore_derivatives as @ignore

import Flux.outputsize
import Flux.trainmode!

# VARIATIONAL FLOWS

abstract type Transform end

@register struct PlanarFlow <: Transform
    w::AbstractArray
    u::AbstractArray
    b::Number
    h::Function
end

function PlanarFlow( dimensions::Int, h::Function=tanh )
    return PlanarFlow( zeros(Float32, dimensions), zeros(Float32, dimensions), Float32(0), h )
end

function (t::PlanarFlow)( z::AbstractVector )
    return z .+ t.u * t.h( t.w ⋅ z + t.b )
end

function ψ(t::PlanarFlow, z::AbstractVector)
    g = gradient( t.h, (t.w ⋅ z + t.b) )[1]
    return g * t.w
end

@register struct Flow
    transforms::Vector{Transform}
end

function Flow( dimensions::Int, length::Int, FlowType::DataType; h::Function=tanh )
    return Flow( [ FlowType( dimensions, h ) for _ in 1:length ] )
end

import Base.eachrow
function Base.eachrow(x::Flux.Zygote.Buffer{Float32, Matrix{Float32}})

    return Iterators.map( Iterators.product( axes(x)[2:end]... ) ) do i

        return x[:, i...]

    end
end

function (flow::Flow)(z::AbstractVector)

    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        z = hcat(z, f(z[:, end]))
    end

    return z[:, end]
end

function (flow::Flow)(z_0::AbstractArray)
    s = size(z_0)
    z = reshape( z_0, s[1], reduce(*, s[2:end] ) )
    f = flow.( z[:, i] for i in 1:size(z, 2) )
    y = hcat( f... )
    return reshape(y, s...)
end

Flux.@functor Flow (transforms, );

# TODO: implement non-log version for precision 
function log_pdf( flow::Flow, q_0::AbstractVector, z_0::AbstractVector )

    z = hcat(z_0, zeros(eltype(z_0), length(z_0), length(flow.transforms)))

    s = 0
    
    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        s += log( 1 + f.u ⋅ ψ(f, z[:, i]) )
        z[:, i+1] = f(z[:, i])
    end

    return log.( q_0 ) .- s

end


# Free-Energy Bound
function FEB( flow::Flow, z::Union{Flux.Zygote.Buffer, AbstractVector} )

    s = 0
    
    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        s += log( 1 + f.u ⋅ ψ(f, z[:, i]) )
        z = hcat( z, f(z[:, end]) )
    end

    return -s
end

# AUTOENCODERS

abstract type AutoEncoder end

function sample_gaussian( μ::T, σ::T ) where T <: Number

    ϵ = rand( Normal(T(0), T(1)) )

    return μ + σ * ϵ

end

function (model::AutoEncoder)(data::AbstractArray)

    @ignore mapreduce( ||, model.flow.layer.transforms ) do transform 

        return transform.w ⋅ transform.u > -1

    end |> println

    E          = model.encoder(data)
    M          = model.μ(E)
    S          = model.σ(E)
    Z          = sample_gaussian.(M, S)
    F          = model.flow(Z)
    Y          = model.decoder(F)

    return (E=E, M=M, S=S, Z=Z, F=F, Y=Y)

end

struct NoNaN <: Flux.Optimise.AbstractOptimiser end

function Flux.Optimise.apply!(o::NoNaN, x, Δ::AbstractArray{T}) where T <: Number

    sanitize(δ)::T = isnan(δ) || isinf(δ) ? T(0) : δ

    Δ = sanitize.(Δ)

end
