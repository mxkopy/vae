include("Util.jl")

using Flux, Flux.NNlib, Distributions, LinearAlgebra, CUDA, Serialization, SpecialFunctions, Printf
using Flux: gradient
using Flux.ChainRulesCore: @ignore_derivatives as @ignore

import Flux.outputsize
import Flux.trainmode!
import LinearAlgebra: dot
import Flux.gpu

# VARIATIONAL FLOWS

abstract type Transform end

@register struct PlanarFlow <: Transform
    w::AbstractVector
    u::AbstractVector
    b::Number
    h::Function
end

function PlanarFlow( dimensions::Int, h::Function=tanh )
    w = ones(dimensions)
    u = zeros(dimensions)
    b = 0
    return PlanarFlow( w, u, b, h )
end

function û(t::PlanarFlow)
    return t.u .+ ( (-1 + log( 1 + exp(t.u ⋅ t.w))) + t.u ⋅ t.w ) .* (t.w ./ (t.w ⋅ t.w))
end

function (t::PlanarFlow)(z::AbstractVector)
    return z + û(t) * t.h( t.w ⋅ z + t.b )
end

function ψ(t::PlanarFlow, z::AbstractVector)
    g, = gradient( t.h, (t.w ⋅ z + t.b) )
    return g * t.w
end

@register struct Flow
    transforms::Vector{Transform}
end

function Flow( dimensions::Int, length::Int, FlowType::DataType; h::Function=tanh )
    return Flow( [ FlowType( dimensions, h ) for _ in 1:length ] )
end

function (flow::Flow)(z::AbstractVector)
    y, s = foldl(flow.transforms, init=(z, 0)) do l, r
        z, c = l
        y = r(z)
        c = c + log( 1 + û(r) ⋅ ψ(r, z) )
        return y, c
    end
    return y, -s
end

function (flow::Flow)(z_0::T) where T <: AbstractArray

    indices = Iterators.product(axes(z_0)[2:end]...)

    z, f = mapreduce(((l_z, l_f), (r_z, r_f)) -> (hcat(l_z, r_z), vcat(l_f, r_f)), indices) do index
        return flow(z_0[:, index...])
    end

    return reshape(z, size(z_0)), reshape(f, size(z_0)[2:end])
end

function (flow::Flow)(z_0::T) where T <: CuArray
    return flow( z_0 |> cpu ) |> gpu
end

Flux.@functor Flow (transforms, );

# AUTOENCODERS

abstract type AutoEncoder end

function sample_gaussian( μ::T, σ::T ) where T <: Number

    ϵ = rand( Normal(T(0), T(1)) )

    return μ + σ * ϵ

end

function (model::AutoEncoder)(data::AbstractArray)

    E          = model.encoder(data)
    M          = model.μ(E)
    S          = model.σ(E)
    Z          = sample_gaussian.(M, S)
    F, FEB     = model.flow(Z)
    Y          = model.decoder(F)

    return (E=E, M=M, S=S, Z=Z, F=F, Y=Y, FEB=FEB)

end

struct NoNaN <: Flux.Optimise.AbstractOptimiser end

function Flux.Optimise.apply!(o::NoNaN, x, Δ::AbstractArray{T}) where T <: Number

    sanitize(δ)::T = isnan(δ) || isinf(δ) ? T(0) : δ

    Δ = sanitize.(Δ)

end
