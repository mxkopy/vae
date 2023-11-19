include("DataIterators.jl")

using HTTP, HTTP.WebSockets

function Base.iterate( channel::Channel, state=nothing )

    if !isopen(channel) return nothing end

    return take!(channel), state

end

function WSClient(
    channel::Channel{Vector{UInt8}};
    host="ws://127.0.0.1", 
    port=ENV["DATA_PORT"]
)
    WebSockets.open( "$host:$port", verbose=true ) do websocket

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
    host::String="ws://127.0.0.1",
    port::Int=ENV["DATA_PORT"]
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
    port::Int=ENV["DATA_PORT"]
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

function DataServer(;
    iterator::BatchIterator=BatchIterator{ImageReader}( ENV["DATA_SOURCE"], 1 ), 
    host::String="0.0.0.0",
    port::Int=ENV["DATA_PORT"]
)
    channel = Channel{Vector{UInt8}}()

    @async WSServer(channel, host=host, port=port)

    for data in iterator

        buf = IOBuffer()

        serialize(buf, data)

        put!(channel, buf.data)

    end

end