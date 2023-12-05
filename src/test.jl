include("Main.jl")

model = ResNetVAE(64)
X = rand(Float32, 32, 32, 3, 1) |> gpu
Flux.gradient( () -> X |> model.encoder |> sum, Flux.params(model.encoder) )