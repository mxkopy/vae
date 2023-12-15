include("DataIterators.jl")

using HTTP, HTTP.WebSockets, JSON

function Base.iterate( channel::Channel, state=nothing )

    if !isopen(channel) return nothing end

    return take!(channel), state

end

function add_info( array::AbstractArray; kwargs... )

    return Dict(

        size   => size(array),
        sizeof => sizeof(array),
        eltype => eltype(array),
        kwargs...

    )

end

function process_raw_image( x::AbstractArray )

    x = permutedims(x, (3, 2, 1, 4))

    x = colorview(RGB, x) .|> RGBA{N0f8}

    x = channelview(x)

    x = reinterpret(UInt8, x)

    x = permutedims(x, (1, 3, 2, 4))

    x = reshape(x, length(x))

    x = Vector{UInt8}(x)

    return x

end

function to_message( objects::Vector{NamedTuple{(:info, :bits), Tuple{Dict{Any, Any}, Vector{UInt8}}}} )::Vector{UInt8}

    metadata_array = []

    offset = 0

    for object in objects

        info  = Dict( "info" => object.info )

        metadata = Dict(

            "range" => Dict( 
                "start" => offset, 
                "end" => offset + sizeof(object.bits) 
            ),
    
            "info" => object.info
        )

        push!(metadata_array, metadata)

        offset += sizeof(object.bits)

    end

    metadata = metadata_array |> JSON.json |> Vector{UInt8}

    payload = mapreduce(t -> t.bits, vcat, objects)

    message = vcat( metadata, [0], payload )

    return message
    
end

function to_message( A::Vector{T} ) where T <: AbstractArray

    return map(A) do x 

        return (info=add_info(x), bits=process_raw_image(x))

    end

end

function WSClient(
    channel::Channel{Vector{UInt8}};
    host="127.0.0.1", 
    port=parse(Int, ENV["DATA_PORT"])
)

    WebSockets.open( "ws://$host:$port", verbose=true ) do websocket

        while !isclosed( websocket ) && isopen( channel )
            try
                data = receive(websocket)
                put!(channel, data)

            catch e

                println(e)
                break

            end

        end

        if !isopen( channel ) close( websocket ) end

    end

end

function DataClient(;
    host::String="127.0.0.1",
    port::Int=parse(Int, ENV["DATA_PORT"])
)
    channel = Channel{Vector{UInt8}}()

    @async WSClient(channel, host=host, port=port)

    return Iterators.map(channel) do message

        return deserialize( IOBuffer(message) )

    end

end

function WSServer(
    channel::Channel{Vector{UInt8}};
    host::String="0.0.0.0",
    port::Int=parse(Int, ENV["DATA_PORT"])
)

    WebSockets.listen( host, port, verbose=true ) do websocket

        while !isclosed( websocket ) && isopen( channel )

            try

                data = take!(channel)
                send(websocket, data)

            catch e

                println(e)
                break

            end
        end

        if !isopen( channel ) close( websocket ) end

    end

end

function DataServer(
    iterator=BatchIterator{ImageReader}( ENV["DATA_TARGET"], 1 );
    host::String="0.0.0.0",
    port::Int=parse(Int, ENV["DATA_PORT"])
)
    channel = Channel{Vector{UInt8}}()

    @async WSServer(channel, host=host, port=port)

    for data in iterator

        buf = IOBuffer()

        serialize(buf, data)

        put!(channel, buf.data)

    end

end
