FROM julia:1.9-bullseye

# set working directory
WORKDIR /VAE

ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia

RUN julia -e 'using Pkg; Pkg.add("CUDA")'
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"11.8")'
RUN julia -e 'using CUDA; CUDA.precompile_runtime()'

COPY src/deps.jl .
RUN julia deps.jl

# copy code into the container
COPY src/ .

CMD []
