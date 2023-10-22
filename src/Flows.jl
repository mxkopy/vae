using Flux, Zygote, Distributions, LinearAlgebra
using Zygote: @ignore 
using Zygote: gradient

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

function (t::PlanarFlow)( z::AbstractVector{T} ) where T <: Number

    return z + t.u * t.h( t.w ⋅ z + t.b ) .|> T

end

function ψ( t::PlanarFlow, z::AbstractVector )

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

function ( flow::Flow )( z_0::AbstractVector{T} ) where T <: Number

    z = z_0

    for f in flow.transforms

        z = f(z)

    end

    return z .|> T

end

function (flow::Flow)( z_0::AbstractArray )

    s = size(z_0)
    
    z = reshape( z_0, size(z_0, 1), reduce(*, size(z_0)[2:end] ) )

    h = [ z[:, i] for i in 1:size(z, 2) ]

    f = flow.( h )

    y = hcat( f... )

    return reshape(y, s...)

end


Flux.@functor Flow (transforms, );

# TODO: implement non-log version for precision 
function log_pdf( flow::Flow, q_0::AbstractVector, z_0::AbstractVector )

    z = z_0

    s = 0

    for t in flow.transforms

        s += log( 1 + t.u ⋅ ψ(t, z) )

        z = t(z)

    end

    return log.( q_0 ) .- s

end


# Free-Energy Bound
function FEB( flow::Flow, z_0::AbstractVector )

    z = z_0

    s = 0

    for t in flow.transforms

        s += log(1 + t.u ⋅ ψ(t, z))

        z = t(z)

    end

    return -s

end

function FEB( flow::Flow, q_0::AbstractArray, z_0::AbstractArray )

    Q = Base.Iterators.product( axes(q_0)[2:end]... )    
    Z = Base.Iterators.product( axes(z_0)[2:end]... )

    E = mapreduce( (l, r) -> hcat(l, r), Q, Z ) do q, z

        return FEB( flow, q_0[:, q...], z_0[:, z...] )

    end

    return reshape(E, size(q_0))

end