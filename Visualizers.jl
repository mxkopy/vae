include("DataIterators.jl")
include("AutoEncoders.jl")
# include("DDSP.jl")
include("ResNet.jl")



function single_visualizer(size=(256, 256))

    gd = imshow( rand(RGB, size) )

    signal = ImageView.Observable( rand(RGB, size) )

    imshow( gd["gui"]["canvas"], signal )

    return function (image)

        signal[] = image
    
    end

end

function grid_visualizer( grid, size=(128, 128) )

    gui = imshow_gui(size, grid) 

    canvases = gui["canvas"]

    coords   = Iterators.product( 1:first(grid), 1:last(grid) ) |> collect

    signals  = map( _ -> ImageView.Observable(rand(RGB, size)), coords)

    for (coord, signal) in zip(coords, signals)

        imshow(canvases[coord...], signal)

    end

    Gtk.showall(gui["window"])

    return function(images) 

        for (signal, image) in zip(signals, images)

            signal[] = image

        end

    end

end

#################
#   ResNetVAE   #
#################


function visualizer(model::ResNetVAE, grid_size=(256, 256))

    # magic = x -> gcd( x / (2^Int(floor(log2(sqrt(x))))) |> floor |> Int, x )

    # model_size = length(model.alpha.bias)

    # grid  = ( magic(model_size), model_size ÷ magic(model_size) )

    # grid_vis, data_vis = grid_visualizer(grid, grid_size), single_visualizer(grid_size)

    # return @noinline function(encoder, decoder)

        # latents = map( 1:length(model.alpha.bias) ) do i

        #     mask = [ idx[1] == i ? 1f0 : 0f0 for idx in CartesianIndices(latent) ] |> model.device

        #     return mask .* latent

        # end

        # images = map( latents ) do L          
            
        #     return model.decoder( L )[:, :, :, 1] |> cpu |> from_color

        # end

        # images = map( 1:length(model.alpha.bias) ) do i

        #     out = encoder[i, :, :, 1]

        #     out = (out .- minimum(out)) ./ (maximum(out) - minimum(out))

        #     return out |> cpu |> A -> RGB.(A, A, A)

        # end

        # grid_vis(images), data_vis(decoder[:, :, :, 1] |> cpu |> from_color)

    # end

    L, R = single_visualizer(grid_size), single_visualizer(grid_size)

    return function(decoder::AbstractArray, data::AbstractArray)

        decoder[:, :, :, 1] |> from_color |> L, data[:, :, :, 1] |> from_color |> R

    end

end



############
#   DDSP   #
############



# function visualizer(model::DDSP)

#     # p1 = plot(1:1, [0])
#     # p2 = plot(1:1, [0])

#     return function(out, data)

#         # x = mean(out,  dims=3)
#         # y = mean(data, dims=3)

#         # x = reshape( x, length(x) )
#         # y = reshape( y, length(y) )

#         # for (o, d) in zip(out, data)

#         #     push!(p1, o)
#         #     push!(p2, d)

#         # end

#         return nothing

#     end

# end
