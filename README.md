# Variational Autoencoders
The latent space of an autoencoder is basically the model's 'internal representation' of some data. It's not a given, though, that this represenation is organized in any meaningful way. 

VAEs solve this problem by regularizing (i.e. smoothing) latent representations according to the Kullback-Leibler distance. This forces data points that are more alike closer together within latent space. And this lets you smoothly interpolate between images, which can be useful for generative tasks.

The reason this works is because the model is learning a latent probability distribution, rather than fixed relationships. By introducing randomness (variations!) into the input of the decoder, the effect of an individual data point is 'smoothed out' in latent space. 

Apparently, some probability distributions are better than others. Hence, I implemented the Dirichlet-VAE from https://arxiv.org/abs/1901.02739. 

# Usage

First, make sure you have all the dependencies installed. deps.sh is a convenience script for this purpose. You'll also want an X server or similar - if you don't know what this is, then you're fine. 

In theory, this repo provides a modally-general library for Dirichlet VAEs. It's agnostic to which specific encoder and decoder you use, as long as the outputs make sense (i.e. they must be arrays of size [model_size, ...]). Subtyping AutoEncoder allows a type to use the AutoEncoder forward-pass, as it has these fields

```
mutable struct $T <: AutoEncoder

  encoder
  decoder
  alpha
  beta
  decode

  precision
  device
    
end

Flux.@functor T (encoder, decoder, alpha, beta, decode)

```

and of course, the output of each field that is a function makes sense. 

You can also use the macro

```
@autoencoder T
```

which creates a subtype T of AutoEncoder, and provides a convenience constructor 

```
T(encoder, decoder, model_size; precision=Float32, device=gpu)
```

which sets the alpha, beta and decode fields to sensible Dense layers. 

In order to train your model, you should define a loss function that looks like this:

```
function loss( model )

  return function lf( data... )
  
    Flux.Losses.Whichever( model(data), data )
  
  end

end
```

If this looks insane, that's because it kind of is. I'm trying to find more elegant ways to visualize a model's output during training, my lack of which makes this indirection necessary. Unfortunately I absolutely slept on macros, which are probably the solution. 

After subtyping AutoEncoder and defining an appropriate loss function, you can populate the 'data/image' directory with images and 'data/audio' for .wav audio. Julia is agnostic to the filesystem used, so you can sshfs or ln -s a COCO training dataset or similar (I use train2017). The DataIterator library stands fairly well alone, and is plug-and-play provided there aren't any formats in the dataset that Julia can't handle (.ogg, .mov, ...). For example, here are the function signatures for some of the high-level BatchIterators:

```
ImageIterator(;directory="data/image/", batches=1, shuffle=false)
AudioIterator(;directory="data/audio/", sample_size=2^16, batches=1, shuffle=false, shuffle_dir=false)
```

The VideoIterator is buggy due to the idiosyncracies of decoding video for arbitrary formats.

---

So, in practice, you should fill data/image, run 

```
julia Main.jl --train
```

and watch the blurry images. Various arguments and switches are listed in Main.jl. Some are not used and I'm debating rewriting the CLI entirely, so I won't bother to go into too much detail. 

---

There are various easter eggs throughout this mess as well, such as an incredibly powered CUDA-compatible broadcast macro, an implemented but unused MelResNet and an Optimiser that simply removes NaN from gradients and weights. Please, find a use for this insanity!
