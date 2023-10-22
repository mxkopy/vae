# FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 as builder
FROM julia:1.9.3-bullseye

ENV JULIA_DEPOT_PATH=/usr/local/share/julia

# set working directory
WORKDIR /VAE

# copy code into the container 
COPY src/ .

# copy data into the container
COPY data data

RUN julia -e 'using Pkg; Pkg.add("CUDA")'
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"12.2")'
RUN julia -e 'using CUDA; CUDA.precompile_runtime()'

RUN julia Main.jl

RUN mkdir -m 0777 /depot
ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia

CMD []