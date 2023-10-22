include("Training.jl")


# model = ResNetVAE( 64 )

model = deserialize("10.18.mdl")["model"]

convert( Float32, model )

opt  = Optimiser( ClipNorm(1f0), ADAM(1f-3, (0.9, 0.99)), NoNaN() )

loss = create_loss_function( model )

trainer = Trainer( model, opt, loss )

images = BatchIterator{ImageReader}( "../data/image", 1 )

for image in images

    trainer( image .|> Float32 )

end


