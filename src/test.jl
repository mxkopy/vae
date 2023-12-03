# 1. creates a constructor that takes into account the precision and device of the part, using the type declarations
# 2. passes through property accesses (i.e. getfield, setfield, getproperty, setproperty) to the underlying type
# 3. composes the declared function call with the function call of the underlying type 
# 4. registers the parameters with Flux

@part struct PermuteInput{T}

    layer::T    

end

function ( part::PermuteInput{T} )(inputs...)



end


macro part( declaration::Expr )

    return eval(:(

        $declaration;

    ))



end
