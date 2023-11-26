using Flux, Serialization, Interpolations, WAV, Zygote, Distributions, CUDA, Images, Printf

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

encoder_block( channels ) = ( downsampler( channels ), channelwise_dense( first(channels) + last(channels) => last(channels) ) )
decoder_block( channels ) = (   upsampler( channels ), channelwise_dense( first(channels) + last(channels) => last(channels) ) )


function resnet_vae_encoder( model_size )

    return Chain(

        x -> unit_normalize.(x),
        encoder_block( 3 => 8 )...,
        encoder_block( 8 => 16 )...,
        encoder_block( 16 => 32 )...,
        encoder_block( 32 => 64 )...,
        encoder_block( 64 => model_size )...
        
    )

end


function resnet_vae_decoder( model_size )

    return Chain(

        decoder_block( model_size => 64 )...,
        decoder_block( 64 => 32 )...,
        decoder_block( 32 => 16 )...,
        decoder_block( 16 => 8 )...,
        decoder_block( 8 => 3 )...,

        x -> unit_denormalize.(x)

    )

end



@autoencoder ResNetVAE

function ResNetVAE( model_size; flow_length=64, flow_type=PlanarFlow, precision=Float32, device=gpu )

    make_channelwise = x -> permutedims(x, (3, 2, 1, 4))

    encoder    = make_channelwise ∘ resnet_vae_encoder(model_size)
    decoder    = resnet_vae_decoder(model_size) ∘ make_channelwise

    m          = Dense( model_size, model_size )
    s          = Dense( model_size, model_size, softplus )

    flow       = Flow( model_size, flow_length, PlanarFlow )

    return ResNetVAE( encoder, decoder, m, s, flow )

end

