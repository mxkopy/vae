include("ResNet.jl")
include("DDSP.jl")

using BSON, ImageView, Gtk, .AutoEncoders, .ResNet, .DDSP, .DataIterators
using .AutoEncoders: sample_dirichlet

# you should probably use these in the REPL

function latent_params( model )

    return function( A, B )

        B = ResNet.interpolate_data(A, B) |> collect

        _, _, a1, b1, l1 = model( A |> DataIterators.preprocess_image .|> model.precision |> model.device )

        _, _, a2, b2, l2 = model( B |> DataIterators.preprocess_image .|> model.precision |> model.device )

        return (a1, b1, l1), (a2, b2, l2)

    end

end

function dirichlet_sampler( model )

    return function(alpha, beta, params=ones( size(alpha) ) .|> model.precision |> model.device )

        return sample_dirichlet( params ./ sum(params, dims=1), alpha, beta )

    end

end


function latent_visualizer( model )

    visualize = ResNet.single_visualizer()

    return function( latent )

        visualize( model.decoder(latent)[:, :, :, 1] |> cpu |> from_color )

    end

end

# example usage
# model      = BSON.load("data/models/image64_1.bson")["model"] 

# latents    = latent_params( model )

# dirichlets = dirichlet_sampler( model )

# visualizer = latent_visualizer( model )

# A, B       = latents( Iterators.take(DataIterators.ImageData(shuffle=true), 2)... )

# visualizer( dirichlets( A[1] .+ B[1], A[2] .+ B[2] ) )

# for dx in range(0, 1f0, 100)

#   visualizer( dirichlets( ((1f0 - dx) .* A[1]) .+ (dx .* B[1]), ((1f0 -dx) .* A[2]) .+ (dx .* B[2]) ) )
#   sleep(0.1)

# end


# function visualize_latent_channels_IMG( vision, grid, size=(128, 128) )

#     model_size = length(vision.alpha.b)

#     vision = vision |> gpu

#     (img,) = Iterators.take(ImageData(shuffle=true), 1) |> collect

#     lat = latent(vision, img |> preprocess_image)

#     visualize = ResNet.grid_visualizer(grid, size)

#     images = map( 1:model_size ) do i

#         mask    = [ idx[1] == i ? 1f0 : 0f0 for idx in CartesianIndices(lat) ] |> gpu

#         decoded = decode(vision, lat .* mask)[:, :, :, 1]

#         return decoded

#     end

#     visualize(images)

# end
