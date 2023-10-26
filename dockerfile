FROM julia:1.8-bullseye
ARG DATA_PATH

# set working directory
WORKDIR /VAE

# copy code into the container 
COPY src/ .

# copy data into the container
COPY $DATA_PATH data

RUN julia -e 'using Pkg; Pkg.add("CUDA")'
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"11.8")'
RUN julia -e 'using CUDA; CUDA.precompile_runtime()'

RUN julia Main.jl

RUN mkdir -m 0777 /depot
ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia

CMD []
