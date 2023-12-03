using Flux, JSON

macro register( struct_declaration::Expr )
    T = struct_declaration.args[2]
    while !(T isa Symbol)
        T = T.args[1]
    end
    return eval(:(
        $struct_declaration;
        function Flux.ChainRulesCore.rrule(x::$T, args...);
            println(x);
            println(args);
        end;
        Flux.@functor $T;
    ))
end

@register struct Device{P <: Number, D}
    layer
    function Device{P, D}(layer) where {P, D}
        function to(x) x end
        function to(x::AbstractArray) x .|> P |> D end
        new{P, D}( fmap(to, layer) )
    end
end
Flux.@functor Device (layer, )

function (d::Device{P, D})(x::AbstractArray) where {P <: Number, D}
    x .|> P |> D |> d.layer
end

function (d::Device{P, D})(x)::P where {P <: Number, D}
    x |> P |> D |> d.layer
end

@register struct PermuteInput
    permutations::NTuple{N, Int} where N
    layer
    PermuteInput(permutations::NTuple, layer) = new(permutations, layer)
    PermuteInput(permutations::NTuple)        = layer -> new(permutations, layer)
    PermuteInput(permutations...)             = layer -> new(permutations, layer)
end
(c::PermuteInput)(data::AbstractArray{T, N}) where {T <: Number, N} = c.layer(permutedims(data, c.permutations))
Flux.@functor PermuteInput (layer, )


@register struct PermuteOutput
    permutations::NTuple{N, Int} where N
    layer
    PermuteOutput(permutations::NTuple, layer) = new(permutations, layer)
    PermuteOutput(permutations::NTuple)        = layer -> new(permutations, layer)
    PermuteOutput(permutations...)             = layer -> new(permutations, layer)
end
(c::PermuteOutput)(data::AbstractArray{T, N}) where {T <: Number, N} = permutedims(c.layer(data), c.permutations)
Flux.@functor PermuteOutput (layer, )


