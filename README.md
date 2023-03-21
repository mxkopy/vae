# Variational Autoencoders
The latent space of an autoencoder is the model's compressed 'internal' representation of some data. It's not a given, though, that this represenation is organized in any meaningful way. 

VAEs solve this problem in two ways: learning a probability distribution rather than a fixed set of points, and by minimizing the Kullback-Leibler divergence between it and a known, smooth distribution. By introducing randomness (variations!) into the decoder's inputs, the effects individual data points are 'smoothed out' in latent space. The KLD, then, constrains them to be regular.

Apparently, some probability distributions are better than others. Hence, I implemented the Dirichlet-VAE from https://arxiv.org/abs/1901.02739. 

# Basic Usage
First, run the following to install the dependencies:

```
./deps.sh
```

Then, make sure you have images in the data/image directory - they will be the training set. It should look something like

```
./data/image/0001.jpg
./data/image/0002.jpg
...
```

Now, you can run:

```
julia Main.jl --train
```

By default, this will train an image model and visualize the output and input data in two GTK windows. 

There are various switches and flags with descriptive names in Main.jl. However, many are unused and the CLI will most likely soon be replaced.

# General Usage
This repo provides a modality-independent library for Dirichlet VAEs. It's agnostic to its encoder and decoder, as long as their outputs make sense (i.e. they must output arrays of size [model_size, ...]). Subtyping AutoEncoder allows the use of the AutoEncoder forward-pass, provided the subtype has at least these fields:

```
struct T <: AutoEncoder

  encoder
  decoder
  alpha
  beta
  interpret
    
end

Flux.@functor T (encoder, decoder, alpha, beta, decode)
```

and of course, the output of each field that is a function makes sense. 

The last line registers the type with Flux, so it can be trained and moved between the gpu and cpu.

You can also use the macro

```
@autoencoder T
```

which does all the above for you, and provides a convenience constructor 

```
T(encoder, decoder, model_size; precision=Float32, device=gpu)
```

that sets the alpha, beta and decode fields to sensible Dense layers.

It's necessary to query the device (cpu or gpu) that the model is on and its precision. If you want to make a custom model, you should implement `query_device(model::T)` and `query_precision(model::T)` since by default it relies on the `model.interpret` field being a Dense layer. 

# Loss Functions
When possible, the methods provided in this library are generic to AutoEncoder (reconstruction_loss is rather domain-specific and doesn't make much sense to generalize). You can define custom behavior by specifying them to your model's type. Here is the list of signatures you can use should you want to do so:

```
create_loss_function(model::T)
elbo_loss(model::T)
visualize_loss(model::T)
print_loss(format::String)
```

And of course, you can add your own. 

The generic create_loss_function composes the rest of these methods in a way similar to this:

```
function create_loss_function( model::AutoEncoder )

  visualize = visualizer(model)
  E         = elbo_loss(model)
  R         = reconstruction_loss(model)

  return function( x::AbstractArray )

    y = model(x)

    e = E(y)
    r = R(y, x)

    visualize( (y,x), (e,r) )

  end

end
```

In general, these methods can persist important state in between iterations, such as a method-of-moments estimator or GUI. This sort of state should be initialized in the outer scope, and can be updated within the returned function.

# Data
After subtyping AutoEncoder and defining an appropriate loss function, you can populate the 'data/image' directory with images and 'data/audio' with .wav audio. Julia is agnostic to the filesystem used, so you can sshfs or ln -s a COCO training dataset or similar (I use train2017). 

The DataIterator library stands fairly well alone, and is plug-and-play provided there aren't any formats in the dataset that Julia can't handle (.ogg, .mov, ...). For example, here are the function signatures for some of the high-level BatchIterators:

```
ImageIterator(;directory="data/image/", batches=1, shuffle=false) -> Array{T, 4}
AudioIterator(;directory="data/audio/", sample_size=2^16, batches=1, shuffle=false, shuffle_dir=false) -> Array{T, 4}
```

# Optimiser
The final ingredient required for a training loop is an optimiser. The default

``` 
Optimiser( ClipNorm(1f0), NoNaN(), ADAM(lr, (0.9, 0.99)) )
```

works well, but it can readily be tuned to your case (see https://fluxml.ai/Flux.jl/stable/training/optimisers/ for inspiration). A word of warning - the functions involved in sampling the Dirichlet distribution can grow explosively, so including the NoNaN() optimiser is highly recommended.  

# Training
Training.jl provides the following function:

```
train( model::AutoEncoder, optimizer::Flux.Optimise.AbstractOptimiser, loss_fn::Function, data::Union{DataIterator, BatchIterator}, filename::String; save_freq=10, epochs=1 )
```
Which is more or less a wrapper around Flux's training loop.

However; if you've gotten this far, you may want to roll your own, which is not difficult at all given a model, loss function, data iterator, and optimiser. See https://fluxml.ai/Flux.jl/stable/training/training/#Training-Loops for an example.
