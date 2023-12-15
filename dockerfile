FROM julia:1.9-bullseye

ENV JULIA_DEPOT_PATH=/depot:/usr/local/share/julia

# RUN julia -e 'using Pkg; Pkg.add("CUDA")'
RUN julia -e 'using Pkg; Pkg.add( ["BSON", "Colors", "Distributions", "FFTW", "FileIO", "Flux", "HTTP", "Images", "ImageTransformations", "Interpolations", "JSON", "LibSndFile", "LinearAlgebra", "NNlib", "Plots", "Printf", "Random", "Serialization", "SliceMap", "SpecialFunctions", "WAV"] )'
RUN julia -e 'using CUDA; CUDA.set_runtime_version!(v"11.8")'
RUN julia -e 'using CUDA; CUDA.precompile_runtime()'

CMD []
