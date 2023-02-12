module DataIterators

export ImageIterator, VideoIterator, AudioIterator, to_color, from_color, preprocess_image, preprocess_audio

using FileIO, VideoIO, Images, ImageTransformations, Colors, WAV, CUDA, Statistics, Random


struct FileReader

    indices::Vector{Int}
    reader
    close_reader

end

Base.length( itr::FileReader ) = length(itr.indices)

Base.isempty( itr::FileReader ) = isempty(itr.indices)


FileReader(reader, close_reader, length::Int; shuffle::Bool=false) = FileReader( shuffle ? Random.shuffle(1:length) : collect(1:length), reader, close_reader )

function Base.iterate( itr::FileReader, indices=itr.indices )

    if !isempty(indices)

        index = popfirst!( indices )

        data  = itr.reader(index)

        return data, indices

    end

    itr.close_reader()

    return nothing

end

function audio_reader(dir, sample_size)

    reader = function(index)

        x    = (index-1) * sample_size + 1 : index * sample_size

        y, _ = wavread(dir, x)

        return y

    end

    close_reader = () -> ()

    return reader, close_reader

end

function video_reader(stream)

    lastindex = 0

    reader = function(index)

        if index < lastindex

            seekstart(stream)

            skipframes(stream, index)
        
        else

            skipframes(stream, index - lastindex - 1)

        end

        lastindex = index

        return read(stream)

    end

    close_reader = () -> close(stream)

    return reader, close_reader

end



function AudioFileReader(dir; sample_size::Int=2^16, shuffle::Bool=false)

    stream = loadstreaming(dir)

    info   = stream.sfinfo

    close(stream)

    reader, close_reader = audio_reader(dir, sample_size)

    return FileReader(reader, close_reader, info.frames ÷ sample_size, shuffle=shuffle)

end

function VideoFileReader(dir; shuffle::Bool=false)

    stream = VideoIO.openvideo(dir)

    reader, close_reader = video_reader(stream)

    return FileReader(reader, close_reader, VideoIO.counttotalframes(stream), shuffle=shuffle)

end

function ImageFileReader(dir; shuffle::Bool=false)

    dirs = readdir(dir, join=true, sort=false)

    dirs = shuffle ? Random.shuffle(dirs) : dirs

    reader, close_reader = index -> load(dirs[index]), () -> ()
    
    return FileReader(reader, close_reader, length(dirs), shuffle=shuffle)

end



struct DataIterator

    files
    shuffle_dir::Bool

end

Base.length(itr::DataIterator)  = sum(file -> length(file), itr.files, init=0)

Base.isempty(itr::DataIterator) = length(itr) == 0

DataIterator(reader, directory::String, shuffle_dir::Bool) = DataIterator( map(reader, readdir(directory, join=true, sort=false) ), shuffle_dir )

# function serialize(itr::DataIterator)

# 



function Base.iterate(itr::DataIterator, files=itr.files)

    if !isempty(files)

        next = itr.shuffle_dir ? rand( axes(files)... ) : firstindex(files)

        if isempty(files[next])

            popat!(files, next)

            return iterate(itr, files)

        end

        data, _ = iterate(files[next])

        return data, files

    end

    return nothing

end


function AudioData(; directory::String="data/audio/", sample_size::Int=2^16, shuffle::Bool=false, shuffle_dir::Bool=false)

    reader = dir -> AudioFileReader(dir, sample_size=sample_size, shuffle=shuffle) 

    return DataIterator(reader, directory, shuffle_dir)

end

function VideoData(; directory::String="data/video/", shuffle::Bool=false, shuffle_dir::Bool=false)

    reader = dir -> VideoFileReader(dir, shuffle=shuffle)

    return DataIterator(reader, directory, shuffle_dir)

end

function ImageData(; directory::String="data/image/", shuffle::Bool=false)

    files = ImageFileReader(directory, shuffle=shuffle)

    return DataIterator([files], false)

end

function to_color( image )

    return image .|> RGB |> channelview

end


function from_color( image )

    return RGB.( image[:, :, 1], image[:, :, 2], image[:, :, 3] )

end

function preprocess_image( image )

    image = image |> to_color
    image = permutedims( image, (2, 3, 1) )
    image = reshape( image, (size(image)..., 1) )

    return image

end


function preprocess_audio(audio)

    return reshape(audio, (size(audio)[1], 1, size(audio)[2], 1))

end



struct BatchIterator

    batches::Int
    iterator
    preprocess

end

Base.length( itr::BatchIterator )  = Base.length(itr.iterator) / itr.batches |> ceil |> Int
Base.isempty( itr::BatchIterator ) = Base.isempty(itr.iterator)


function Base.iterate(itr::BatchIterator, iterator=itr.iterator)

    data = Iterators.take(iterator, itr.batches)

    op   = (l, r) -> cat(l, r, dims=4)

    return isempty(iterator) ? nothing : (mapreduce(itr.preprocess, op, data), iterator)

end



function AudioIterator(;directory="data/audio/", sample_size=2^16, batches=1, shuffle=false, shuffle_dir=false)

    iterator = AudioData(directory=directory, sample_size=sample_size, shuffle=shuffle, shuffle_dir=shuffle_dir)

    return BatchIterator(batches, iterator, preprocess_audio)

end


function VideoIterator(;directory="data/video/", batches=1, shuffle=false, shuffle_dir=false)

    iterator = VideoData(directory=directory, shuffle=shuffle, shuffle_dir=shuffle_dir)

    return BatchIterator(batches, iterator, preprocess_image)

end


function ImageIterator(;directory="data/image/", batches=1, shuffle=false)

    iterator = ImageData(directory=directory, shuffle=shuffle)

    return BatchIterator(batches, iterator, preprocess_image)

end


# function save_data( itr::BatchIterator )


# end

# function load_data( )


end
