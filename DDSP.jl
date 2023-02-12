module DDSP

export AudioVAE

include("AutoEncoders.jl")
include("DataIterators.jl")

using Flux, NNlib, FFTW, SliceMap, CUDA, Plots, Printf, .AutoEncoders, .DataIterators

unit_normalize   = x -> x #x -> (x / 2f0) + 5f-1
unit_denormalize = hardtanh #x -> (x * 2f0) - 1f0

activation       = tanh
nonlinearity     = x -> x #relu





function encoder(model_size)


end

function decoder(model_size)


end




function spectral_distance( out, data; ϵ=1f-8 )

    F  = a -> abs.(a) .^ 2 |> sum |> sqrt
    L1 = a -> abs.(a)      |> sum 

    Y = fft(out,  [1])
    X = fft(data, [1])

    A = F(X .- Y) / max( F(X), ϵ )
    B = log( L1( X .- Y ) )

    return A + B

end



function loss( out, data, alpha, alpha_parameter )
    
    elbo    = ELBO(out, alpha, alpha_parameter)

    s_loss  = spectral_distance(out, data)

    r_loss  = Flux.Losses.mse(out, data)

    Zygote.ignore() do 

        @printf "\nr_loss %.5e s_loss %.5e -elbo %.5e" r_loss s_loss -elbo
        flush(stdout)

    end

    return r_loss + s_loss #* 1f-2 - elbo * 1f-2

end

function loss(model)

    return nothing

end


function visualizer(model)

    # p1 = plot(1:1, [0])
    # p2 = plot(1:1, [0])

    return function(out, data)

        # x = mean(out,  dims=3)
        # y = mean(data, dims=3)

        # x = reshape( x, length(x) )
        # y = reshape( y, length(y) )

        # for (o, d) in zip(out, data)

        #     push!(p1, o)
        #     push!(p2, d)

        # end

        return nothing

    end

end


function AudioVAE(model_size; device=gpu)

    enc = encoder(model_size)
    dec = decoder(model_size)

    return AutoEncoder( enc, dec, model_size, device=device )

end

end

include("SomeMacros.jl")

using FFTW, Flux



mel = h -> 1125 * log( 1 + h/700 )
hz  = m -> 700  * ( exp( m/1125 ) - 1 )

function rectangular_window(L, x)

    return 1

end


function hann_window(L, x)

    sin(pi * x / L)^2

end

function smoothing_window(L, x)

    return 3*L / (2*L^2-1) * (1 - (2*(x+1)/L-1)^2 )

end

function triangular_window(L, x)

    return 1 - abs(2*x/L - 1)

end



struct Window

    f::Function

end

function (window::Window)( y; l=1, r=first(size(y)) )

    x = (l - 2) .+ first(axes(y))

    w = similar(y, length(x))

    w .= window.f.((r - l), x)

    return w .* y

end

Guard( window::Function ) = (N, n) -> 0 <= n && n <= N ? window(N, n) : 0

RectangularWindow = Window( Guard( rectangular_window ) )
HannWindow        = Window( Guard( hann_window       ) )
SmoothingWindow   = Window( Guard( smoothing_window  ) )
TriangularWindow  = Window( Guard( triangular_window ) )


pad(x::AbstractVector; l=length(x)÷2, r=l) = NNlib.pad_zeros(x, (l, r))
pad(x::AbstractVector, n; offset=0) = pad(x, l=offset, r=n-offset-length(x))




function centered_strides(x::AbstractVector; stride=length(x), l=-Inf, r=Inf, offset=0, bound=length(x) )

    return map( offset:stride:bound ) do i 

        L, R = max(l+i, firstindex(x)), min(r+i, lastindex(x))

        return x[L:R]

    end

end


frames(x::AbstractVector; stride=length(x), len=stride, offset=0) = centered_strides(x, stride=stride, l=1, r=len, offset=offset, bound=length(x)-len)




# stride / len = 1 - overlap
# function windowed_fft(x::AbstractVector; window=HannWindow, stride=length(x) ÷ 2, len=length(x))
function windowed_fft(x::AbstractVector; window=HannWindow, stride=length(x) ÷ 2, len=2^8 )

    return mapreduce(+, enumerate(frames(x, stride=stride, len=len))) do (i, x̄)

        X = window(x̄) |> fft

        return pad(X, length(x), offset=(i-1)*len)

    end

end


function filter_overlap_add(h::AbstractVector, x::AbstractVector; len=256)

    return mapreduce( +, enumerate(frames(x; stride=len)) ) do (i, x̄)

        H = pad( h, r=length(x̄))

        X = pad( x̄, r=length(h)) |> fft

        Y = ifft( H .* X ) .|> real

        return pad( Y, length(x), offset=(i-1)*len )

    end

end



function mel_spectrogram(x::AbstractVector; window=HannWindow, len=1024, stride=256, n_filters=229, sample_rate=44100, min_f=300, max_f=8000, ϵ=1f-8 )

    lerp = k -> (mel(max_f) - mel(min_f)) * (k / (n_filters + 2)) + mel(min_f)

    bin  = f -> (len + 1) * (f / sample_rate) |> floor |> Int

    filter_pt = bin ∘ hz ∘ lerp

    return mapreduce( vcat, frames(x, stride=stride, len=len) ) do x̂

        F = x̂ |> window |> fft

        F = abs.(F) .^ 2 ./ len 

        return mapreduce( hcat, 1:n_filters ) do k

            f = TriangularWindow(F, l=filter_pt(k-1), r=filter_pt(k+1))

            return sum(f) 

        end

    end

end


function A_weighting(f)

    return (12194^2 * f^4) / ( (f^2 + 20.6^2) * sqrt((f^2 + 107.7^2) * (f^2 + 737.9^2)) * (f^2 + 12194^2) )

end


function loudness(x::AbstractVector; sr=44100, stride=2048)

    return mapreduce(vcat, frames(x, stride=stride) ) do x̄

        return mapreduce(+, enumerate(x̄ |> fft)) do (i, X̄)
        
            return A_weighting( i * sr / stride ) * abs(X̄) / stride |> log

        end

    end

end


function MelResNet(sample_rate=44100)

    function residual_block( channels, stride )

        in, out = first(channels), last(channels)

        pool    = stride == 1 ? identity : MeanPool( (1, stride), pad=SamePad() )

        dim     = in == out ? identity : Conv( (1, 1), in => out )

        return dim ∘ SkipConnection( Chain(

            BatchNorm( in,     relu ), Conv( (1, 1),     in => in ÷ 4,                     pad=SamePad() ),
            BatchNorm( in ÷ 4, relu ), Conv( (3, 3), in ÷ 4 => in ÷ 4, stride=(1, stride), pad=SamePad() ),
            BatchNorm( in ÷ 4, relu ), Conv( (1, 1), in ÷ 4 => in,                         pad=SamePad() ),

        ), (fx, x) -> fx + pool(x) )

    end

    return Chain(

        A -> mel_spectrogram(A, sample_rate=sample_rate), 

        Conv( (7, 7), 2 => 64, stride=(1, 2), pad=SamePad()),
 
        MaxPool( (1, 3), stride=(1, 2), pad=SamePad() ), 
    
        residual_block(   64 =>  128, 1 ),
        residual_block(  128 =>  128, 1 ),
        residual_block(  128 =>  256, 2 ),
        residual_block(  256 =>  256, 1 ),
        residual_block(  256 =>  256, 1 ),
        residual_block(  256 =>  512, 2 ),
        residual_block(  512 =>  512, 1 ),
        residual_block(  512 =>  512, 1 ),
        residual_block(  512 =>  512, 1 ),
        residual_block(  512 => 1024, 2 ),
        residual_block( 1024 => 1024, 1 ),
        residual_block( 1024 => 1024, 1 ), 

        A -> reshape(A, ( first(size(A)), :, last(size(A)) )),

        A -> permutedims(A, (2, 1, 3) ),  

        Dense( 1024 * 8, 128 ), 

        A -> permutedims(A, (2, 1, 3) ), 

        A -> reshape(A, (first(size(A)), 1, 128, :) ), 

        ConvTranspose( (512, 1), 128 => 128, stride=(8, 1) ),
        AdaptiveMeanPool( (1000, 1) ),

        A -> softplus.(A), 

        A -> Flux.normalise(A, dims=1)

    )

end
