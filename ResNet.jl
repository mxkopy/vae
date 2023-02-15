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

function downsampler( channels; stride=2, kernel=3 )

    down = MeanPool( (stride, stride), pad=SamePad() )

    SkipConnection(

        conv_block( (kernel, kernel), channels, stride=stride, type=Conv ),
        (fx, x) -> cat(down(x), fx, dims=3)
        
    )

end

function upsampler( channels; stride=2, kernel=3 )

    up = Upsample( stride )

    SkipConnection(

        conv_block( (kernel, kernel), channels, stride=stride, groups=1, type=ConvTranspose ),
        (fx, x) -> cat(up(x), fx, dims=3)
        
    )

end

function resnet_vae_encoder( model_size )

    return Chain(

        x -> unit_normalize.(x),
        downsampler( 3  => 8 ),            channelwise_dense( 11 => 8 ),
        downsampler( 8  => 16 ),           channelwise_dense( 24 => 16 ),
        downsampler( 16 => 32 ),           channelwise_dense( 48 => 32 ),
        downsampler( 32 => 64 ),           channelwise_dense( 96 => 64 ),
        downsampler( 64 => model_size ),   channelwise_dense( 64 + model_size => model_size )
        
    )

end


function resnet_vae_decoder( model_size )

    return Chain(

        upsampler( model_size => 64 ),  channelwise_dense( model_size + 64 => 64 ),
        upsampler( 64 => 32 ),          channelwise_dense( 96 => 32 ),
        upsampler( 32 => 16 ),          channelwise_dense( 48 => 16 ),
        upsampler( 16 => 8 ),           channelwise_dense( 24 => 8 ),
        upsampler( 8 => 3 ),            channelwise_dense( 11 => 3 ), 
        x -> unit_denormalize.(x)

    )

end



@autoencoder ResNetVAE

ResNetVAE( model_size::Int; device=gpu ) = ResNetVAE( resnet_vae_encoder(model_size), resnet_vae_decoder(model_size), model_size, device=device )
