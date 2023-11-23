# Variational Autoencoders
The latent space of an autoencoder is the model's compressed 'internal' representation of some data. It's not a given, though, that this represenation is organized in any meaningful way. 

VAEs solve this problem in two ways: learning a probability distribution rather than a fixed set of points, and by minimizing the Kullback-Leibler divergence between it and a known, smooth distribution. By introducing randomness (variations!) into the decoder's inputs, the effects of individual data points are 'smoothed out' in latent space. The KLD, then, constrains them to be regular.

# Basic Usage
Run

```
DATA_SOURCE=/images/to/train/on docker-compose up data training frontend
```

By default, the visualizer is served at http://127.0.0.1:2998/frontend.html . 

# General Usage
This repo provides a modality-independent library for VAEs. It's agnostic to its encoder and decoder, as long as their outputs make sense (i.e. they must output arrays of size [model_size, ...]). Subtyping AutoEncoder allows the use of the AutoEncoder forward-pass, provided the subtype has at least these fields:

```
struct T <: AutoEncoder

  encoder
  decoder
  μ
  σ
  flow
    
end

Flux.@functor T (encoder, decoder, μ, σ, flow)
```

and of course, the output of each field that is a function makes sense. 

The last line registers the type with Flux, so it can be trained and moved between the gpu and cpu.

You can also use the macro

```
@autoencoder T
```

which does all the above for you.

# Loss Functions
When possible, the methods provided in this library are generic to AutoEncoder. You can define custom behavior by specifying them to your model's type. Here is the list of signatures you can use should you want to do so:

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

In general, these methods can persist important state in between iterations, such as a method-of-moments estimator or a connection to a server. This sort of state should be initialized in the outer scope, and can be updated within the returned function.

# Data
For now, the data service container uses data on the host machine pointed to by the DATA_SOURCE environment variable.

The DataIterator library stands fairly well alone, and is plug-and-play provided there aren't any formats in the dataset that Julia can't handle (.ogg, .mov, ...).

# Optimiser
The final ingredient required for a training loop is an optimiser. The default

``` 
Optimiser( ClipNorm(1f0), NoNaN(), ADAM(lr, (0.9, 0.99)) )
```

works well, but it can readily be tuned to your case (see https://fluxml.ai/Flux.jl/stable/training/optimisers/ for inspiration).

