using Flux, Serialization, Interpolations, WAV, Distributions, CUDA, Images, Printf

using Flux.ChainRulesCore: @ignore_derivatives as @ignore

unit_normalize   = x -> x 
unit_denormalize = hardtanh ∘ relu

activation       = tanh

function conv_block( kernel, channels; type=Conv, σ=activation, stride=1, groups=gcd(channels...), norm_layer=true )

    return type( kernel, channels, σ, stride=stride, groups=groups, pad=SamePad() )

    # return Chain(

        # type( kernel, channels, σ, stride=stride, groups=groups, pad=SamePad() ),

        # norm_layer ? GroupNorm( last(channels), gcd(channels...) ) : x -> x

        # norm_layer ? BatchNorm( last(channels) ) : x -> x

    # )

end

function channelwise_dense( channels; init=Flux.identity_init )

    return Dense( channels..., init=init ) |> PermuteInput(3, 1, 2, 4) |> PermuteOutput(2, 3, 1, 4)

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

@register struct ResNetVAE <: AutoEncoder
    encoder
    decoder
    μ
    σ
    flow
    ResNetVAE(args...; precision=Float32, device=gpu) = new( (args .|> Device{precision, device})... )
end

function ResNetVAE( model_size; flow_length=64, flow_type=PlanarFlow, precision=Float32, device=gpu )

    encoder    = resnet_vae_encoder(model_size) |> PermuteOutput(3, 1, 2, 4)
    decoder    = resnet_vae_decoder(model_size) |> PermuteInput(2, 3, 1, 4)

    μ          = Dense( model_size, model_size )
    σ          = Dense( model_size, model_size, softplus )

    flow       = Flow( model_size, flow_length, PlanarFlow )

    return ResNetVAE(encoder, decoder, μ, σ, flow, precision=precision, device=device)

end