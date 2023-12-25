function from_message( message ){

    let metadata_end = message.findIndex(x => x == 0);

    let metadata_buffer = message.slice(0, metadata_end);

    let metadata_string = new TextDecoder().decode(metadata_buffer);
    
    let metadata = JSON.parse(metadata_string);

    let payload  = message.slice(metadata_end + 1);

    let objects = [];

    for( let { info, range: {start, end} } of metadata ){

        let data = payload.slice(start, end);

        objects.push( {...info, data: data} );

    }

    return objects;

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

    async on_message(event){

        let message = new Uint8ClampedArray( await event.data.arrayBuffer() );

        let objects = from_message(message);

        for( const {name, data, height, width} of objects ){

            let canvas = this.canvases[name];

            let ctx = canvas.getContext('2d');
        
            let img = new ImageData( data, height, width, {colorSpace: 'display-p3'} );

            ctx.clearRect( 0, 0, canvas.height, canvas.width );
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
        this.ws.addEventListener( "message", this.on_message )
        this.ws.addEventListener( "close", console.log )

    }

    disconnectedCallback(){

        this.observer.disconnect();

    }

}

customElements.define( "video-stream", Stream );