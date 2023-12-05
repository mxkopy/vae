include("Util.jl")

using Flux, Distributions, LinearAlgebra, CUDA, Serialization, SpecialFunctions, Printf
using Flux: gradient
using Flux.ChainRulesCore: @ignore_derivatives as @ignore

import Flux.outputsize
import Flux.trainmode!

# VARIATIONAL FLOWS

const default_init = (x...) -> rand(Uniform(-1f-2, 1f-2), x...)

abstract type Transform end

struct PlanarFlow <: Transform
    w::AbstractArray
    u::AbstractArray
    b::Number
    h::Function
end

function PlanarFlow( dimensions::Int, init::Function=default_init, h::Function=tanh )
    return PlanarFlow( init(dimensions), init(dimensions), init(1)..., h )
end

function (t::PlanarFlow)( z::AbstractVector )
    return z .+ t.u * t.h( t.w ⋅ z + t.b )
end

function ψ(t::PlanarFlow, z::AbstractVector)
    g = gradient( t.h, (t.w ⋅ z + t.b) )[1]
    return g * t.w
end

Flux.@functor PlanarFlow (w, u, b, h);

struct Flow
    transforms::Vector{Transform}
end

function Flow( dimensions::Int, length::Int, FlowType::DataType; init::Function=default_init, h::Function=tanh )
    return Flow( [ FlowType( dimensions, init, h ) for _ in 1:length ] )
end

function ( flow::Flow )( z_0::AbstractVector )

    z = zeros(eltype(z_0), (length(z_0), length(flow.transforms)+1))
    z[:, 1] = z_0
    z = Flux.Zygote.Buffer(z)

    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        z[:, i+1] = f(z[:, i])
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

    s = (length(z_0), length(flow.transforms)+1)
    z = similar(z_0, dims=s)
    z[:, 1] = z_0
    z = Flux.Zygote.Buffer(z)

    s = 0
    
    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        s += log( 1 + f.u ⋅ ψ(f, z[:, i]) )
        z[:, i+1] = f(z[:, i])
    end

    return log.( q_0 ) .- s
end


# Free-Energy Bound
function FEB( flow::Flow, z_0::Union{Flux.Zygote.Buffer, AbstractVector} )

    z = zeros(eltype(z_0), (length(z_0), length(flow.transforms)+1))
    z[:, 1] .= z_0
    z = Flux.Zygote.Buffer(z)

    s = 0
    
    for i in 1:length(flow.transforms)
        f = flow.transforms[i]
        s += log( 1 + f.u ⋅ ψ(f, z[:, i]) )
        z[:, i+1] = f(z[:, i])
    end

    return -s
end

function FEB( flow::Flow, q_0::AbstractArray{P}, z_0::AbstractArray{P} ) where P <: Number 

    Q = Base.Iterators.product( axes(q_0)[2:end]... )    
    Z = Base.Iterators.product( axes(z_0)[2:end]... )

    E = mapreduce( (l, r) -> hcat(l, r), Q, Z ) do q, z

        return FEB( flow, q_0[:, q...], z_0[:, z...] )

    end

    return reshape(E, size(q_0))

end




# AUTOENCODERS

abstract type AutoEncoder end

function sample_gaussian( μ::T, σ::T ) where T <: Number

    ϵ = @ignore rand( Normal(T(0), T(1)) )

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
