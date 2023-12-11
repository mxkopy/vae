# include("Main.jl")
# model = ResNetVAE(64)
# M = model
# X = rand(N0f8, 256, 256, 3, 1) .|> Float32 |> gpu 
# loss = ReconstructionLoss(model) 
# Y = [ Flux.gradient( () -> loss(X, M(X)[:Y]), Flux.params(M) )... ]
include("Main.jl")
using BenchmarkTools

gpu_model = ResNetVAE(64, device=gpu)
cpu_model = ResNetVAE(64, device=cpu)

X = rand(Float32, 256, 256, 3, 1) |> gpu; @btime gpu_model(X)
X = rand(Float32, 256, 256, 3, 1) |> cpu; @btime cpu_model(X)