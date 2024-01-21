include("Main.jl")
using BenchmarkTools
model = ResNetVAE(64, device=gpu)
x = rand(Float32, 256, 256, 3, 1) |> gpu
loss  = create_loss_function(model)

# g = gradient( Flux.params(model) ) do 
#     loss(x)
# end

e = model.encoder(x)
μ = model.μ(e)
σ = model.σ(e)
z = sample_gaussian.(μ, σ)
f, feb = model.flow(z)
y = model.decoder(f)

@btime model.encoder(x)
@btime model.μ(e)
@btime model.σ(e)
@btime sample_gaussian.(μ, σ)
@btime model.flow(z)
@btime model.decoder(f)
@btime loss(x)

# Z_gpu         = [ rand(Float32, 64) |> gpu for _ in 1:64 ]
# Z_cpu         = [ rand(Float32, 64)        for _ in 1:64 ]
# cpu_model     = ResNetVAE(64, device=cpu)
# gpu_model     = ResNetVAE(64)
# transform_cpu = cpu_model.flow.layer.transforms[1]
# transform_gpu = gpu_model.flow.layer.transforms[1]

# @btime Z_cpu[1] .+ û(transform_cpu) .* t.h( transform_cpu.w ⋅ Z_cpu[1] + transform_cpu.b )
# @btime Z_gpu[1] .+ û(transform_gpu) .* t.h( transform_gpu.w ⋅ Z_gpu[1] + transform_gpu.b )

# @btime transform_cpu.w ⋅ Z_cpu[1] + transform_cpu.b
# @btime transform_gpu.w ⋅ Z_gpu[1] + transform_gpu.b

# @btime transform_cpu.w ⋅ Z_cpu[1]
# @btime transform_gpu.w ⋅ Z_gpu[1]


