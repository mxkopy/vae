# Variational Autoencoders
The latent space of an autoencoder is the model's compressed 'internal' representation of some data. It's not a given, though, that this represenation is organized in any meaningful way. 

VAEs solve this problem by regularizing (i.e. smoothing) latent representations by minimizing the Kullback-Leibler distance between them. This clusters data points that are similar together in latent space, which lets you smoothly interpolate between them, opening up some fun generative possibilities.

This works because the model is learning a latent probability distribution, rather than fixed relationships/points. By introducing randomness (variations!) into the input of the decoder, the effect of an individual data point is 'smoothed out' in latent space. 

Apparently, some probability distributions are better than others. Hence, I implemented the Dirichlet-VAE from https://arxiv.org/abs/1901.02739. 

# Basic Usage
First, make sure you have all the dependencies installed. deps.sh is a convenience script for this purpose. You'll also want an X server or similar for visualizations - if you don't know what this is, then you're fine. 

Then, make sure you have images in the data/image folder - they will be the training set.

Now, you can run:

```
julia Main.jl --train
```

By default, this will train an image model and visualize the output and input data in two GTK windows. 

There are various switches and flags with descriptive names in Main.jl. However, many are unused and the CLI will most likely soon be deprecated.

# General Usage
This repo provides a modality-independent library for Dirichlet VAEs. It's agnostic to its encoder and decoder, as long as their outputs make sense (i.e. they must output arrays of size [model_size, ...]). Subtyping AutoEncoder allows the use of the AutoEncoder forward-pass, provided the subtype has at least these fields:

```
mutable struct T <: AutoEncoder

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

The last line registers the type with Flux, so it can be trained and moved between the gpu and cpu.

You can also use the macro

```
@autoencoder T
```

which creates a subtype T of AutoEncoder, and provides a convenience constructor 

```
T(encoder, decoder, model_size; precision=Float32, device=gpu)
```

that sets the alpha, beta and decode fields to sensible Dense layers. 

# Loss Function
In order to train your model, you should define a loss function that looks something like this:

```
function loss( model )

  return function lf( data )
  
    Flux.Losses.Whichever( model(data), data )
  
  end

end
```

Sometimes, it's helpful to persist state between iterations (e.g. schedulers, visualizers, etc.), which is why we have this nested abomination. I plan to change this.

Note that the ELBO estimate (which is KL distance plus some other stuff) isn't included in this template, so you'll have to add that as well if you want regularized latent spaces. I also plan to change this.

# Data
After subtyping AutoEncoder and defining an appropriate loss function, you can populate the 'data/image' directory with images and 'data/audio' with .wav audio. Julia is agnostic to the filesystem used, so you can sshfs or ln -s a COCO training dataset or similar (I use train2017). 

The DataIterator library stands fairly well alone, and is plug-and-play provided there aren't any formats in the dataset that Julia can't handle (.ogg, .mov, ...). For example, here are the function signatures for some of the high-level BatchIterators:

```
ImageIterator(;directory="data/image/", batches=1, shuffle=false) -> Array{T, 4}
AudioIterator(;directory="data/audio/", sample_size=2^16, batches=1, shuffle=false, shuffle_dir=false) -> Array{T, 4}
```

I do not recommend using VideoIterator, as video is weird. Video readers don't persist quite as nicely as directory readers.

# Optimiser
The final ingredient required for a training loop is an optimiser. The default

``` 
Optimiser( ClipNorm(1f0), NoNaN(), ADAM(lr, (0.9, 0.99)) )
```

works well, but it can easily be tuned to your case (see https://fluxml.ai/Flux.jl/stable/training/optimisers/ for inspiration). A word of warning - the functions involved in sampling the Dirichlet distribution can grow explosively, so including the NoNaN() optimiser is highly recommended.  

# Training
Training.jl provides the following function:

```
train( model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser, loss_fn::Function, data::Union{DataIterator, BatchIterator}, filename::String; save_freq=10, epochs=1 )
```
Which is more or less a wrapper around Flux's training loop.

However; if you've gotten this far, you may want to roll your own, which is not difficult at all given a model, loss function, data iterator, and optimiser. See https://fluxml.ai/Flux.jl/stable/training/training/#Training-Loops for an example.

# Playing Around With It
AutoEncoderOutputs.jl provides a set of functions that make it easier to visualize manipulations in latent space. A sample script is commented out at the bottom. For now, it's expressely for the ResNetVAE.

# Etc
This was a learning project more than anything for me, so I really went all over the place. If manipulating multidimensional arrays is something you do often, I recommend checking out SomeMacros.jl, which provides a very powerful CUDA-compatible broadcast macro. I found it to be useful in implementing DDSP.
