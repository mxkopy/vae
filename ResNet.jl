using Flux, Serialization, Interpolations, WAV, Zygote, Distributions, CUDA, Images, ImageView, Gtk, Printf

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



function resnet_vae_encoder( model_size )

    encoder      = pre_encoder( model_size )

    reshaper     = x -> permutedims( x, (3, 2, 1, 4) )

    return Chain( 
        
        x -> unit_normalize.(x), 
        encoder...,
        reshaper

    ) 

end



function resnet_vae_decoder( model_size )

    decoder       = post_decoder( model_size )

    reshaper      = x -> permutedims(x, (3, 2, 1, 4))

    return Chain( 
        reshaper, 
        decoder..., 
        x -> unit_denormalize.(x) 
    )

end

@autoencoder ResNetVAE

ResNetVAE( model_size::Int; device=gpu ) = ResNetVAE( resnet_vae_encoder(model_size), resnet_vae_decoder(model_size), model_size, device=device )

