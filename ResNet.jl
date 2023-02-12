module ResNet

export ResNetVAE

include("AutoEncoders.jl")
include("DataIterators.jl")

using Flux, Serialization, Interpolations, WAV, Zygote, Distributions, CUDA, Images, ImageView, Gtk, Printf, .AutoEncoders, .DataIterators

using Zygote: @ignore

unit_normalize   = x -> x #x -> x * 2f0 - 1f0
unit_denormalize = hardtanh ∘ relu  #x -> 5f-1 * tanh(x) + 5f-1

activation       = tanh

function conv_block( kernel, channels; type=Conv, σ=activation, stride=1, groups=gcd(channels...), norm_layer=true )

    return Chain(

        type( kernel, channels, σ, stride=stride, groups=groups, pad=SamePad() ),

        # norm_layer ? GroupNorm( last(channels), gcd(channels...) ) : x -> x

        # norm_layer ? BatchNorm( last(channels) ) : x -> x

    )

end

function channelwise_dense( channels; init=Flux.identity_init )

    return Chain(

        A -> permutedims(A, (3, 2, 1, 4)), 

        Dense( channels..., init=init ), 

        A -> permutedims(A, (3, 2, 1, 4))

    )

end

function downsampler( channels; stride=2, kernel=3, dense_channels=false )

    down = MeanPool( (stride, stride), pad=SamePad() )

    return Chain(

        SkipConnection(

            conv_block( (kernel, kernel), channels, stride=stride, type=Conv ),
            (fx, x) -> cat(down(x), fx, dims=3)
         
        ),

        dense_channels ? 

        channelwise_dense( last(channels) + first(channels) => last(channels) ) :

        Conv( ( 1, 1 ), last(channels) + first(channels) => last(channels), pad=SamePad() )

    )

end



function upsampler( channels; stride=2, kernel=3, dense_channels=false )

    up = Upsample( stride )

    return Chain(

        SkipConnection(

            conv_block( (kernel, kernel), channels, stride=stride, type=ConvTranspose ),
            (fx, x) -> cat(up(x), fx, dims=3)
         
        ),

        dense_channels ? 

        channelwise_dense( last(channels) + first(channels) => last(channels) ) :

        ConvTranspose( ( 1, 1 ), last(channels) + first(channels) => last(channels), pad=SamePad() )

    )

end



function inception( channels; stride=1, type=Conv, connect=(x...) -> reduce( (l, r) -> cat(l, r, dims=3), x) )

    conv1a   = conv_block((1, 1), channels, σ=relu, type=type)
    conv1b   = conv_block((1, 1), channels, σ=relu, type=type)
    conv1c   = conv_block((1, 1), channels, σ=relu, type=type)

    conv3a   = conv_block((3, 3), last(channels) => last(channels), σ=relu, stride=stride, type=type)
    conv3b_1 = conv_block((3, 3), last(channels) => last(channels), σ=relu)
    conv3b_2 = conv_block((3, 3), last(channels) => last(channels), σ=relu, stride=stride, type=type)

    pool   = stride > 1 ? type == Conv ? MeanPool( (stride, stride), pad=SamePad() ) : Upsample(stride) : x -> x

    A      = Chain( pool, conv1c )
    B      = Chain( conv1b, conv3b_1, conv3b_2 )
    C      = Chain( conv1a, conv3a )

    return Chain( 
        
        Parallel( connect, A, B, C ),

        (connect == +) ? x -> x : channelwise_dense( last(channels) * 3 => last(channels) )

    )

end



function pre_encoder( out_channels )

    return Chain(

        downsampler( 3  => 8, dense_channels=true ),
        downsampler( 8  => 16, dense_channels=true ),
        downsampler( 16 => 32, dense_channels=true ),
        downsampler( 32 => 64, dense_channels=true ),
        downsampler( 64 => out_channels, dense_channels=true ),
        
        # ResNet( 64 => out_channels, stride=1 ),

    )

end


function post_decoder( in_channels )

    return Chain(

        # ResNet( in_channels => 64, stride=1 ),

        upsampler( in_channels => 64, dense_channels=true ),
        upsampler( 64 => 32, dense_channels=true ),
        upsampler( 32 => 16, dense_channels=true ),
        upsampler( 16 => 8, dense_channels=true ),
        upsampler( 8 => 3, dense_channels=true ) 

    )

end



function encoder( model_size )

    encoder      = pre_encoder( model_size )

    reshaper     = x -> permutedims( x, (3, 2, 1, 4) )

    return Chain( 
        
        x -> unit_normalize.(x), 
        encoder...,
        reshaper

    ) 

end



function decoder( model_size )

    decoder       = post_decoder( model_size )

    reshaper      = x -> permutedims(x, (3, 2, 1, 4))

    return Chain( 
        reshaper, 
        decoder..., 
        x -> unit_denormalize.(x) 
    )

end


function interpolate_data(out, data)

    mode = map( o -> o == 1 ? NoInterp() : BSpline(Linear()), size(out) )

    axes = map( ((d, o),) -> range(1, d, o), zip( size(data), size(out) ) )

    itp  = interpolate(data, mode)

    return itp( axes... )

end

# this creates a loss function that's used in training
# includes stuff like visualizers & scheduling
# i don't imagine super abstract usage so for now many things are hardcoded

# schedule_period is a pair giving the number of iterations for the annealing & fixed loss, respectively
# from https://arxiv.org/pdf/1903.10145.pdf

function loss( model; α_true=fill(0.98, length(model.alpha.bias)) .|> model.precision |> model.device, burn_in=10000, schedule_period=400=>250 )

    visualize = visualizer(model)

    schedule  = vcat(range(0.0, 1f-3, first(schedule_period)), fill(1.0, last(schedule_period))) |> Iterators.cycle |> Iterators.Stateful

    mme, n    = alpha_mme(α_true), 0

    return @noinline function(data)

        encoder, decoder, α, β, latent = model(data)

        data    = @ignore interpolate_data(decoder, data) .|> model.precision |> model.device
        
        w       = @ignore first(schedule)

        α_t     = @ignore mme(latent)

        α_true  = @ignore (n += 1) > burn_in ? α_t : α_true

        elbo    = ELBO(decoder, α, α_true)

        r_loss  = Flux.Losses.mse(decoder, data)

        Zygote.ignore() do  
    
            visualize(data, decoder)
    
            @printf "\nr_loss %.5e -elbo %.5e %i" r_loss -elbo n
            flush(stdout)
    
        end

        return r_loss - elbo * 1f-2

    end

end



function single_visualizer(size=(256, 256))

    gd = imshow( rand(RGB, size) )

    signal  = ImageView.Observable( rand(RGB, size) )

    imshow( gd["gui"]["canvas"], signal )

    return function (image)

        signal[] = image
        # ImageView.push!(signal, image)
    
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

            # ImageView.push!(signal, image)

        end

    end

end



function visualizer(model, grid_size=(256, 256))

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

    return function(decoder, data)

        decoder[:, :, :, 1] |> cpu |> from_color |> L, data[:, :, :, 1] |> cpu |> from_color |> R

    end

end


function ResNetVAE( model_size; device=gpu )

    return AutoEncoder( encoder(model_size), decoder(model_size), model_size, device=device )

end


end
