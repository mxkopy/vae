include("Losses.jl")

using BSON, ImageView, Gtk

# you should probably use these in the REPL

function latent_space_sampler( model::AutoEncoder )

    return function( images::Vararg{AbstractArray} )

        min_h = minimum( img -> size(img, 1), images )

        min_w = minimum( img -> size(img, 2), images )

        return map( images ) do image

            image = interpolate_data(image, (min_h, min_w, 3, 1))

            return model(image)[:latent]

        end

    end

end

function dirichlet_sampler()

    return function( x::AbstractArray{T}, alpha=ones(T, size(x)), beta=alpha ) where T

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
# model      = BSON.load("data/models/image1024.bson")["model"] 

# latents    = latent_space_sampler( model )

# vis        = latent_visualizer( model )

# dirichlets = dirichlet_sampler()

# A, B       = latents( Iterators.take(ImageIterator(shuffle=true), 2)...)

# vis( dirichlets( A .+ B, A .+ B ) )

# for dx in range(0, 1f0, 100)

#   vis( dirichlets( ((1f0 - dx) .* A) .+ (dx .* B), ((1f0 -dx) .* A) .+ (dx .* B) ) )
#   sleep(0.1)

# end
