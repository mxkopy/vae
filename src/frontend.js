async function to_img( payload, h, w ){

    let pixels = new Uint8ClampedArray( payload.length + payload.length / 3 );

    let i = 0;
    let k = 0;

    while( i < h * w ){

        pixels[ 0 + i * 4 ] = payload[ k + 0 * h * w ];
        pixels[ 1 + i * 4 ] = payload[ k + 1 * h * w ];
        pixels[ 2 + i * 4 ] = payload[ k + 2 * h * w ];
        pixels[ 3 + i * 4 ] = 255;

        i++;
        k++;

    }

    return new ImageData( pixels, h, w );

}

class Stream extends HTMLElement {

    constructor(){

        super();

        this.on_mutation = this.on_mutation.bind(this);
        this.on_message  = this.on_message.bind(this);

        this.canvases = {};
    
    }

    on_mutation( mutations, observer ){

        for( const mutation of mutations){

            for( const node of mutation.addedNodes ){

                if( node.nodeName == 'CANVAS' ){

                    this.canvases[node.getAttribute('name')] = node;

                }

            }

            for( const node of mutation.removedNodes ){

                delete this.canvases[node.getAttribute('name')];

            }

        }

        observer.disconnect();

    }

    async on_message( event ){

        let message = await event.data.arrayBuffer();

        let payload = new Uint8ClampedArray( message );

        let metadata_end = payload.findIndex( x => x == 0 );
    
        let metadata_string = new TextDecoder().decode( payload.slice(0, metadata_end) );
    
        let metadata = JSON.parse( metadata_string );

        let i = metadata_end + 1;

        for( const name of Object.keys(metadata) ){

            let size = metadata[name].size;

            let length = size.reduce( (l, r) => l * r, 1 );

            let data = payload.slice(i, i + length + 1);

            i += length;

            let ctx = this.canvases[name].getContext('2d');

            let img = await to_img( data, size[0], size[1] );

            ctx.clearRect( 0, 0, this.canvases[name].height, this.canvases[name].width );
            ctx.putImageData( img, 0, 0 );

        }
    
    }    

    connectedCallback(){

        this.observer = new MutationObserver(this.on_mutation);

        this.observer.observe( this, {
            childList: true,
            subtree: true
        })

        const host = this.getAttribute('host');
        const port = this.getAttribute('port');

        this.ws = new WebSocket( `ws://${host}:${port}` );

        this.ws.addEventListener( "open", console.log )
        // this.ws.addEventListener( "message", event => event.data.arrayBuffer().then( this.on_message ) )
        this.ws.addEventListener( "message", this.on_message )
        this.ws.addEventListener( "close", console.log )

    }

    disconnectedCallback(){

        this.observer.disconnect();

    }

}

customElements.define( "video-stream", Stream );