include("Losses.jl")

using BSON, ImageView, Gtk

# you should probably use these in the REPL

function latent_space_sampler( model::AutoEncoder )

    return function( images::Vararg{AbstractArray} )

        maxsize = argmax( image -> reduce(*, image |> size), images ) |> size

        return map( images ) do image

            image = interpolate_data(image, maxsize)

            return model(image)[:latent]

        end

    end

end

function dirichlet_sampler( model::AutoEncoder )

    return function( x, alpha=ones( size(alpha) ), beta=ones( size(alpha) ) )

        x = convert(model, x ./ sum(x, dims=1) ) # ensures x is in the support of the dirichlet distribution

        return sample_dirichlet( x, alpha, beta )

    end

end


function latent_visualizer( model::AutoEncoder )

    visualize = single_visualizer()

    return function( latent::AbstractArray )

        interpret = model.interpret(latent)
        interpret = permutedims(interpret, (3, 2, 1, 4))

        visualize( model.decoder(interpret)[:, :, :, 1] |> from_color )

    end

end

# example usage
# model      = BSON.load("data/models/image64.bson")["model"] 

# latents    = latent_space_sampler( model )

# dirichlets = dirichlet_sampler( model )

# vis        = latent_visualizer( model )

# A, B       = latents( Iterators.take(ImageData(shuffle=true), 2)... )

# vis( dirichlets( A[1] .+ B[1], A[2] .+ B[2] ) )

# for dx in range(0, 1f0, 100)

#   vis( dirichlets( ((1f0 - dx) .* A[1]) .+ (dx .* B[1]), ((1f0 -dx) .* A[2]) .+ (dx .* B[2]) ) )
#   sleep(0.1)

# end
