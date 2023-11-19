using FileIO, VideoIO, Images, ImageTransformations, Colors, WAV, CUDA, Statistics, Random, LibSndFile, HTTP, HTTP.WebSockets, Serialization
using HTTP.WebSockets: isclosed



abstract type FileReader end;

struct ImageReader <: FileReader

    indices::Vector{String}

end

struct AudioReader <: FileReader

    indices::Vector{Tuple{String, Int}}
    sample_size::Int

end

function Base.iterate( reader::FileReader, state=reader )

    if !isempty( reader )

        index = popfirst!( reader.indices )

        data  = read( reader, index )

        return data, state

    end

    return nothing

end

function read( reader::AudioReader, index::Tuple{String, Int} )

    d = first(index)

    x = last(index)

    s = reader.sample_size

    r = (x - 1) * s + 1 : x * s

    y = wavread(d, r) |> first

    y = reshape( y, (size(y)[1], 1, size(y)[2], 1) )

    return y

end

function read( reader::ImageReader, index::String )

    y = load(index)

    y = y .|> RGB |> channelview

    y = permutedims( y, (2, 3, 1) )

    y = reshape( y, (size(y)..., 1) )

    return y

end

function filter_by_extension( directories::Vector{String}, accepted_extensions::Vector{String} )

    return filter( directories ) do directory

        return mapreduce( x -> !isnothing(x), |, match.(Regex.(accepted_extensions), directory) )

    end

end

function ImageReader( directory::String; accepted_extensions=[".jpg"] )

    directories = readdir(directory, join=true, sort=false)
    directories = filter_by_extension( directories, accepted_extensions )

    return ImageReader( directories )

end

function AudioReader( directory::String, sample_size::Int; accepted_extensions=[".wav"] )

    indices::Vector{Tuple{String, Int}} = []

    directories = readdir( directory, join=true, sort=false )
    directories = filter_by_extension( directories, accepted_extensions )

    for directory in directories

        stream = loadstreaming(directory)

        info   = stream.sfinfo
    
        close(stream)
    
        for i in 1:info.frames ÷ sample_size

            push!( indices, (directory, i) )

        end
    
    end

    return AudioReader( indices, sample_size )

end

Base.length( reader::FileReader ) = length(reader.indices)
Base.isempty( reader::FileReader ) = isempty(reader.indices)

struct BatchIterator{ReaderType}

    reader::ReaderType
    batches::Int

end

function Base.iterate( iterator::BatchIterator, state=iterator.reader )

    if !isempty(iterator)

        data = Iterators.take(iterator.reader, iterator.batches)
    
        return reduce((l, r) -> cat(l, r, dims=4), data), iterator.reader

    end

    return nothing

end

function BatchIterator{AudioReader}( directory::String, sample_size::Int, batches::Int )

    return BatchIterator( AudioReader( directory, sample_size ), batches )
    
end

function BatchIterator{ImageReader}( directory::String, batches::Int )

    return BatchIterator( ImageReader( directory ), batches )

end


Base.copy( iterator::T ) where T <: Union{BatchIterator, FileReader} = T( [ copy( getfield(iterator, field) ) for field in fieldnames(T) ]... )


Base.length( iterator::BatchIterator ) = ceil( length(iterator.reader) / iterator.batches ) 
Base.isempty( iterator::BatchIterator ) = isempty( iterator.reader )
Base.isopen( iterator::BatchIterator ) = !Base.isempty( iterator )