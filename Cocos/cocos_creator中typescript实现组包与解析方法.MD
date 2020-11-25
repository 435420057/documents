# cocos_creator中typescript实现 二进制 组包与解析方法

CocosCreator中以TypeScript操作二进制通常我们需要使用到ArrayBuffer,以及各类型数组如：Uint8Array, Int8Array等,也能满足我们日常的需求，但是操作没有像后端程序(如:Go)那样方便快捷. 其中由于ArrayBuffer是定长的，所以需要事先分配好长度。不能向Golang那个样可以自动增长，合并两个ArrayBuffer也没有那样方便，通常我们在与服务器通信的时候都需要对二进制数据进行组包，虽然ArrayBuffer可以通过DataView来实现数据读取与写入，但是它对int64,uint64这样的数据是不支持的。

这里为了解决这些问题，我将对ArrayBuffer,DataView进行分装,提供两个操作二进制的类型，分别实现数据的读取与写入。

#支持高低字节序#

### 数据读取入类ReadBuffer

支持int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64以及二进制数组的写入。

这里还要强调一点，由于在部分浏览器并不支持int64、uint64大数字，这里我们需要引入第三方库来帮我们统一解决这个问题。

第三方库：https://github.com/kawanet/int64-buffer

```ts
import {Int64BE, Int64LE, Uint64BE, Uint64LE} from "int64-buffer"

export default class ReadBuffer {
    _pos:number;
    _buffer:ArrayBuffer;
    _view:DataView;
    _littleEndian:boolean;

    constructor(buffer:ArrayBuffer, littleEndian?:boolean){
        this._pos = 0;
        this._buffer = buffer;
        this._view = new DataView(buffer);
        this._littleEndian = littleEndian;
    } 
    
    public readInt8():number {
        let n:number = this._view.getInt8(this._pos);
        this._pos+= Int8Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readUint8():number {
        let n:number = this._view.getUint8(this._pos);
        this._pos+= Uint8Array.BYTES_PER_ELEMENT;
        return n;
    }


    public readInt16():number {
        let n:number = this._view.getInt16(this._pos, this._littleEndian);
        this._pos+= Int16Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readUint16():number {
        let n:number = this._view.getUint16(this._pos, this._littleEndian);
        this._pos+= Uint16Array.BYTES_PER_ELEMENT;
        return n;
    }


    public readInt32():number {
        let n:number = this._view.getInt32(this._pos, this._littleEndian);
        this._pos+= Int32Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readUint32():number {
        let n:number = this._view.getUint32(this._pos, this._littleEndian);
        this._pos+= Uint32Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readInt64():number {
        if(this._littleEndian) {
            return new Int64LE(this.readBytes(8)).toNumber();
        }

        return new Int64BE(this.readBytes(8)).toNumber();
    }

    public readUint64():number {
        if(this._littleEndian) {
            return new Uint64LE(this.readBytes(8)).toNumber();
        }

        return new Uint64BE(this.readBytes(8)).toNumber();
    }

    public readFloat32():number {
        let n:number = this._view.getFloat32(this._pos, this._littleEndian);
        this._pos+= Float32Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readFloat64():number {
        let n:number = this._view.getFloat64(this._pos, this._littleEndian);
        this._pos+= Float64Array.BYTES_PER_ELEMENT;
        return n;
    }

    public readBytes(size:number):ArrayBuffer {
        if(this._pos + size > this._buffer.byteLength) {
            return new ArrayBuffer(0);
        }
        let n:ArrayBuffer = this._buffer.slice(this._pos, this._pos + size);
        this._pos+=size;
        return n;
    }
    
    public bytes():ArrayBuffer {
        return this._buffer.slice(this._pos, this._buffer.byteLength);
    }

}
```

具体用方法：

```ts
import ReadBuffer from "./Bytes/ReadBuffer"; // 引入ReadBuffer

let buffer:ReadBuffer = new ReadBuffer(data); // data 为Arraybuffer数据

console.log(buffer.readInt8());
```

这里 bytes 方法返回没有读取完的数据,以方便其它方法进行继续读取


### 数据写入类WriteBuffer

WriteBuffer 类实现与ReadBuffer对应的写功能.

```ts
import {Int64BE, Int64LE, Uint64BE, Uint64LE} from "int64-buffer"

export default class WriteBuffer {
    _pos:number;
    _buffer:ArrayBuffer;
    _view:DataView;
    _littleEndian:boolean;

    constructor(buffer?:ArrayBuffer, littleEndian?:boolean){
        if(buffer == undefined) {
            buffer = new ArrayBuffer(1024);
        } 

        this._pos = 0;
        this._buffer = buffer;
        this._view = new DataView(buffer);
        this._littleEndian = littleEndian;
    } 

    public writeInt8(v:number):void {
        this.mybeExpand(Int8Array.BYTES_PER_ELEMENT);
        this._view.setInt8(this._pos, v);
        this._pos+= Int8Array.BYTES_PER_ELEMENT;
    }

    public writeUint8(v:number):void {
        this.mybeExpand(Uint8Array.BYTES_PER_ELEMENT);
        this._view.setUint8(this._pos, v);
        this._pos += Uint8Array.BYTES_PER_ELEMENT;
    }

    public writeInt16(v:number):void {
        this.mybeExpand(Int16Array.BYTES_PER_ELEMENT);
        this._view.setInt16(this._pos, v, this._littleEndian);
        this._pos += Int16Array.BYTES_PER_ELEMENT;
    }

    public writeUint16(v:number):void {
        this.mybeExpand(Uint16Array.BYTES_PER_ELEMENT);
        this._view.setUint16(this._pos, v, this._littleEndian);
        this._pos += Uint16Array.BYTES_PER_ELEMENT;
    }

    public writeInt32(v:number):void {
        this.mybeExpand(Int32Array.BYTES_PER_ELEMENT);
        this._view.setInt32(this._pos, v, this._littleEndian);
        this._pos += Int32Array.BYTES_PER_ELEMENT;
    }

    public writeUint32(v:number):void {
        this.mybeExpand(Uint32Array.BYTES_PER_ELEMENT);
        this._view.setUint32(this._pos, v, this._littleEndian);
        this._pos += Uint32Array.BYTES_PER_ELEMENT;
    }

    public writeInt64(v:number):void {
        let data:ArrayBuffer;

        if (this._littleEndian) {
            data = new Int64LE(v).toArrayBuffer();
        } else {
            data = new Int64BE(v).toArrayBuffer();
        }
    
        let bigInt8:Int8Array = new Int8Array(data);
        for(var i= 0; i < bigInt8.length; i ++) {
            this.writeInt8(bigInt8[i]);
        }
    }

    public writeUint64(v:number):void {
        let data:ArrayBuffer;

        if (this._littleEndian) {
            data = new Uint64LE(v).toArrayBuffer();
        } else {
            data = new Uint64BE(v).toArrayBuffer();
        }
    
        let bigInt8:Uint8Array = new Uint8Array(data);
        for(var i= 0; i < bigInt8.length; i ++) {
            this.writeUint8(bigInt8[i]);
        }
    }

    public writeFloat32(v:number):void {
        this.mybeExpand(Float32Array.BYTES_PER_ELEMENT);
        this._view.setFloat32(this._pos, v, this._littleEndian);
        this._pos += Float32Array.BYTES_PER_ELEMENT;
    } 

    public writeFloat64(v:number):void {
        this.mybeExpand(Float64Array.BYTES_PER_ELEMENT);
        this._view.setFloat64(this._pos, v, this._littleEndian);
        this._pos += Float64Array.BYTES_PER_ELEMENT;
    }
    
    public writeBytes(v:ArrayBuffer):void {
        let newBuffer:ArrayBuffer = new ArrayBuffer(this._buffer.byteLength + v.byteLength);
        let frame:Uint8Array = new Uint8Array(newBuffer);

        frame.set(new Uint8Array(this._buffer), 0);
        frame.set(new Uint8Array(v), this._pos);
        this._pos += v.byteLength;
        this._buffer = newBuffer;
        this._view = new DataView(this._buffer);
    }

    public bytes():ArrayBuffer {
        return this._buffer.slice(0, this._pos);
    }

    private mybeExpand(length:number) {
        if(this._pos + length < this._buffer.byteLength) {
            return;
        }

        let newBuffer:ArrayBuffer = new ArrayBuffer(this._buffer.byteLength + length);
        let frame:Uint8Array = new Uint8Array(newBuffer);
        frame.set(new Uint8Array(this._buffer), 0)

        this._buffer = newBuffer;
        this._view = new DataView(this._buffer);
    }
}
```

WriteBuffer 可以传入一个由用户指定长度的buffer,如果没有使用将默认创建一个1024长度buffer。bytes方法负责返回写入的内容，如果内容没有buffer的指定长度那么长。bytes只返回用户写入内容，写多少算多少，不会全部返回buffer.

另外，如果写入数据超出了buffer的长度，那么程序会自动扩容。


### 完整使用方法

```ts
import ReadBuffer from "./Bytes/ReadBuffer";
import WriteBuffer from "./Bytes/WriteBuffer";

let body:ArrayBuffer = new ArrayBuffer(2);
let bodyView:DataView = new DataView(body);
bodyView.setInt8(111);
bodyView.setInt8(-111);


let w:WriteBuffer = new WriteBuffer();
w.writeInt8(1);
w.writeUint32(2);
w.writeInt16(3);
w.writeInt16(4);
w.writeBytes(body);


let r:ReadBuffer = new ReadBuffer(w.bytes());
console.log(r.readInt8());
console.log(r.readUInt32());
console.log(r.readInt16());
console.log(r.readInt16());
console.log(r.bytes());
```
