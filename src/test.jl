include("Main.jl")
model = ResNetVAE(64)
M = model
X = rand(N0f8, 256, 256, 3, 1) .|> Float32 
loss = ReconstructionLoss(model) 
Y = [ Flux.gradient( () -> loss(X, M(X)[:Y]), Flux.params(M) )... ]