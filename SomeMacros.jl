using Base.Iterators: product

# broadcasts a function call over selected axes

# @broadcast F(args...) arg1[select1....], arg2[select2....] ... (output_idx_order...)

# example: 
# @broadcast f(x, y) x[i, :, j] y[i, :] (i, j, 2, 1)

# the above command execute f(x[i, :, j], y[i, :, y3]) where i, j, y3 iterate through axes(x, 1) == axes(y, 1), axes(x, 3), axes(y, 3)
# note that the selectors are padded. this means that if z has shape (10, 20, 30), then selecting z[:] will iterate over z[:, i, k] - this can be very costly.

# if a selector sn doesn't satisfy axes(x, n) == axes(y, n), then a warning will be printed but the program will continue

# the last argument of the macro is a tuple specifying the permututations of the output dimensions. 
# numbers specify the position of the corresponding function output 
# selectors specify the position of the corresponding broadcast dimension
# colons are interpreted as the remaining axes of the broadcast output, in order, after numbers and selectors

# e.g: 

# size(x) == (10)
# size(y) == (20)
# size( f( x[i], y[j] ) == (3, 5) )

# then, if output_idx_order == (1, j, 2, i)
# then the size of the macro's output would be (3, 20, 5, 10)

# this is also padded, so output_idx_order == (2, i)
# gives the output of size (5, 10, 3, 20)

function process_selectors( X::Vararg{Union{Expr, Symbol}} )

    return map( X ) do selector

        if selector isa Symbol

            indices = map( i -> Symbol("$selector$i"),  1:ndims(eval(selector)) )

            return Expr( :ref, selector, indices...)

        end

        if selector isa Expr && selector.head === :ref

            x = popfirst!(selector.args)

            indices = map( i -> Symbol("$x$i"),  length(selector.args):ndims(eval(x))-1 )

            return Expr(:ref, x, selector.args..., indices...)

        end

        if selector isa Expr && selector.head === :tuple

            return selector

        end

    end

end


function process_arguments( F::Expr, X::Vararg{Expr} )

    for I in eachindex( F.args )

        if F.args[I] isa Expr && F.args[I].head === :kw 

            if F.args[I].args[1] in ( x.args[1] for x in X )

                @eval $(F.args[I].args[1]) = $(F.args[I].args[2])
                
                F.args[I] = F.args[I].args[1]

            end

        end

    end

    return F

end


function unique_axes( S::Vector{Symbol}, I::Vector{Vector{Any}}; warn=true )

    indices, axises = [], []
    
    for (s, i) in zip(S, I)

        for (c, idx) in enumerate(i)
            
            if !(idx === :(:) || idx isa Number)

                axis = axes(eval(s), c)

                if warn && idx in indices && axises[findfirst(isequal(idx), indices)] != axis

                    println("[ Warning ] @broadcast: $idx has inconsistent axes.")

                end

                push!(indices, idx)
                push!(axises, axis)

            end

        end

    end

    return indices, axises

end


function broadcast_indices( S::Vector{Symbol}, I::Vector{Vector{Any}} )

    indices, axises = unique_axes(S, I)

    return map( product( axises... ) ) do coordinates

        return map( I ) do i

            return map( i ) do idx

                if idx in indices

                    return coordinates[ findfirst(isequal(idx), indices) ]

                end

                return eval(idx)

            end

        end

    end

end




function broadcast_parameters( F::Expr, X::Tuple, I::Vector, isvec::Bool )

    return map( F.args ) do P 

        L = findfirst( x -> P === x.args[1], X )

        if isnothing(L)

            return P

        else

            x = :( view($P, $(I[L])...) )

            return isvec ? :( vec( eval($x) ) ) : x

        end

    end

end


# (1, i, j, 2)

function reconstitute_dims( Y::AbstractArray, S::Vector{Symbol}, I::Vector{Vector{Any}}, O::Expr )

    indices, axises = unique_axes(S, I)

    Y = reshape(Y, axises..., axes(Y)[1:end-1]...)

    C = 1:ndims(Y) |> collect

    permutation = []

    for O in vcat(O.args, fill( (:), ndims(Y) - length(O.args) ))

        i = first( C )

        if O in indices

            i = findfirst(isequal(O), indices)

        elseif O isa Number

            i = length(axises) + O - 1

        end

        push!(permutation, popat!(C, findfirst(isequal(i), C)))

    end

    return permutedims( Y, permutation )


end


function selectors_and_indices( X::Vararg{Expr} )

    I = [ x.args for x in X if x.head === :ref ]

    S = popfirst!.( I )

    return S, I

end


function broadcast( F::Expr, X::Vararg{Expr} )

    S, I = selectors_and_indices( X... )

    O = findfirst( x -> x.head === :tuple, X)

    f = popfirst!(F.args)

    D = broadcast_indices(S, I)

    Y = mapreduce(vcat, D) do D

        P = broadcast_parameters(F, X, D, count( isequal(:(:)), D ) == 1 )

        y = @eval $f( $(P...) )

        return y isa Number ? y : reshape(y, 1, size(y)...)
        
    end

    O = isnothing(O) ? :( () ) : X[O]

    Y = reconstitute_dims(Y, S, I, O)

    return Y

end


macro broadcast( F::Expr, X::Vararg{Expr}=F.args[2] )

    X = process_selectors(X...)

    F = process_arguments(F, X...)

    return broadcast(F, X...)

end