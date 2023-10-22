include("DataIterators.jl")

using HTTP, HTTP.WebSockets

const T = Dict( "Float16" => Float16, "Float32" => Float32, "Float64" => Float64, "N0f8" => N0f8 )

# Connects to a server to retrieve data and put it into a buffer
# By using a buffer, we can create an iterator that waits for data
struct Connection{T}

    name::String
    host::String
    buffer::Channel{T}

end

function Base.iterate( connection::Connection, state=nothing )

    if !isopen(connection.buffer) return nothing end

    return take!(connection.buffer), state

end

Base.close( connection::Connection ) = close( connection.buffer )

# n_frames determines how many frames should be received from the server before running a function F on the received data 
# i.e. first receive metadata, then receive payload -> n_frames=2
function Connection(;

    name::String="0", 
    host="ws://127.0.0.1", 
    port=2999, 
    buffer_size::Int=0, 
    receiver::Function=default_receiver
)

    connection = Connection( name, host, Channel(buffer_size) )

    @async WebSockets.open( "$host:$port" ) do websocket

        send( websocket, name )

        while !isclosed( websocket ) && isopen( connection.buffer )

            try
            
                data = receive( websocket ) |> receiver
                put!( connection.buffer, data )

            catch e

                println(e)
                break

            end

        end

        if !isopen( connection.buffer ) close( websocket ) end

    end

    return connection

end

struct DataServer

    server::HTTP.Server

end

function DataServer(;

    host::String="127.0.0.1",
    port::Int=2999,
    iterator=BatchIterator( ImageReader("/VAE/data/image"), 1 ),
    sender::Function=default_sender,
    save::Bool=true

)

    WebSockets.listen( host, port ) do websocket

        name = receive( websocket )

        while !isclosed( websocket ) && isopen( iterator )

            try

                data = first( iterator )
                sender( websocket, data )

            catch e

                println(e)

                if save
                    serialize(  "data/models/" * name * ".checkpoint", iterator )
                end

                break

            end
        end

        if !isopen( iterator ) close( websocket ) end

    end

    # return DataServer( server )

end


function default_receiver( data )

    i = findfirst( [ UInt8('\n') ], data ) |> first

    metadata, payload = data[1:i], data[i+1:end]

    type, data_size = split( metadata |> String, ";" )

    return reshape( reinterpret( T[type], payload ), [ parse( Int, s ) for s in split(data_size) ]... )

end


function default_sender( websocket, array )

    payload  = reinterpret( UInt8, reshape(array, reduce(*, size(array))) )

    metadata = Vector{UInt8}( "$(eltype(array));$(reduce(*, "$s " for s in size(array)))\n" )

    data = vcat( metadata, payload )

    send( websocket, data )

end



function filter_by_extension( directories::Vector{String}, accepted_extensions::Vector{String} )

    return filter( directories ) do directory

        return mapreduce( x -> !isnothing(x), |, match.(Regex.(accepted_extensions), directory) )

    end

end
