function to_img( payload, h, w ){

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

    return new ImageData( pixels, h );

    // return new ImageData( pixels, h, w, {colorSpace: 'display-p3'} )

}

class Stream extends HTMLElement {

    constructor(){
        super();
    }

    connectedCallback(){

        const shadow = this.attachShadow({ mode: "open" });

        const host = this.getAttribute('host');
        const port = this.getAttribute('port');

        this.ws = new WebSocket( `ws://${host}:${port}` );

        this.canvas = document.createElement('canvas');

        this.canvas.setAttribute('height', this.getAttribute('height'))
        this.canvas.setAttribute('width',  this.getAttribute('width'))

        const on_message = event => {

            event.data.arrayBuffer().then( buf => {

                let data = new Uint8ClampedArray( buf );

                let i = data.findIndex( x => x == ';'.charCodeAt(0) );
                let k = data.findIndex( x => x == '\n'.charCodeAt(0) );
    
                let typestr = new TextDecoder().decode( data.slice(0, i) );
                let sizestr = new TextDecoder().decode( data.slice(i+1, k) );
    
                let size     = sizestr.split(' ');
                let payload  = data.slice(k+1);
    
                const [h, w] = [ Number(size[0]), Number(size[1]) ]

                console.log(sizestr, ' ', payload.length)
    
                // let ctx      = this.canvas.getContext('2d');
                // let img      = to_img( payload, h, w );
    
                // ctx.clearRect( 0, 0, this.canvas.height, this.canvas.width );
                // ctx.putImageData( img, 0, 0 );

            })

        }

        // this.ws.addEventListener( "message", console.log )

        this.ws.addEventListener( "open", console.log )
        this.ws.addEventListener( "message", on_message )
        this.ws.addEventListener( "close", console.log )

        shadow.appendChild( this.canvas );

    }

}

customElements.define("video-stream", Stream );