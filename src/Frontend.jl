using JSON, HTTP

function substitute_environment_variables( text )

    for var in eachmatch( r"\$(.*)\s", text )

        variable = replace( var.match, r"\s" => "" ); 

        value = ENV["$(var |> first)"]

        text = replace( text, variable => value )

    end

    return text

end


function frontend_html( request::HTTP.Request )

    frontend_html = open( "$(ENV["SRC_TARGET"])/frontend.html" ) |> Base.read |> substitute_environment_variables

    return HTTP.Response( 200, ["Accept" => "text/html"], frontend_html )

end



function frontend_js( request::HTTP.Request )

    frontend_js = open( "$(ENV["SRC_TARGET"])/frontend.js" ) |> Base.read

    return HTTP.Response( 200, ["Accept" => "text/javascript"], frontend_js )

end


const router = HTTP.Router()
HTTP.register!( router, "/frontend.html", frontend_html )
HTTP.register!( router, "/frontend.js", frontend_js )
