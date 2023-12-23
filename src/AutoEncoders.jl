include("Util.jl")

using Flux, Flux.NNlib, Distributions, LinearAlgebra, CUDA, Serialization, SpecialFunctions, Printf
using Flux: gradient
using Flux.ChainRulesCore: @ignore_derivatives as @ignore

import Flux.outputsize
import Flux.trainmode!
import LinearAlgebra: dot

# VARIATIONAL FLOWS

abstract type Transform end

@register struct PlanarFlow <: Transform
    w::AbstractVector
    u::AbstractVector
    b::Number
    h::Function
end

function PlanarFlow( dimensions::Int, h::Function=tanh )
    init = (x...) -> rand( Normal(Float32(0), Float32(1)), x... )
    w = init(dimensions)
    u = init(dimensions)
    b = init(1) |> first
    return PlanarFlow( w, u, b, h )
end

function u_hat(w::T, u::T, û::T) where {T <: AbstractVector}

    û .= u .+ ( (-1 + log( 1 + exp(u ⋅ w))) + u ⋅ w ) .* (w ./ (w ⋅ w))

    return nothing

end

function run_flow(w::T, u::T, b::N, h::F, z::A) where {T <: AbstractVector, N <: Number, F <: Function, A <: AbstractMatrix}

    for i in axes(z, 2)

        @inbounds z[:, i] .+= u #.* h(w .* z[:, i] |> sum)

    end

    return nothing

end

function (t::PlanarFlow)(z::AbstractMatrix)

    û = similar(t.u)

    @cuda u_hat(t.w, t.u, û)

    @cuda run_flow(t.w, t.u, t.b, t.h, z)

    return z
    
end

function ψ(t::PlanarFlow, z::AbstractVector)
    g, _ = gradient( t.h, (t.w ⋅ z + t.b) )
    return g * t.w
end

@register struct Flow
    transforms::Vector{Transform}
end

function Flow( dimensions::Int, length::Int, FlowType::DataType; h::Function=tanh )
    return Flow( [ FlowType( dimensions, h ) for _ in 1:length ] )
end

function (flow::Flow)(z::AbstractVector) where T

    return foldl((l, r) -> r(l), flow.transforms, init=reshape(z, length(z), 1))

end

function (flow::Flow)(z_0::AbstractArray)
    s = size(z_0)
    Z = [ z_0[:, i...] for i in Iterators.product(axes(z)[2:end]...) ]

    for transform in flow.transforms

        Z = map(transform, Z )

    end

    Z = reduce(hcat, Z)

    return reshape(Z, s...)
end

Flux.@functor Flow (transforms, );

# Free-Energy Bound
function FEB( flow::Flow, z::Union{Flux.Zygote.Buffer, AbstractVector} )

    _, s = foldl( flow.transforms, init=(z, 0) ) do l, r

        z = l[1]
        c = l[2]

        return r(z), c + log( 1 + û(r) ⋅ ψ(r, z) )

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
