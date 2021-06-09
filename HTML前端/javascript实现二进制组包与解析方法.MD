# javascript实现二进制组包与解析方法

javascript操作二进制通常我们需要使用到ArrayBuffer,以及各类型数组如：Uint8Array, Int8Array等,也能满足我们日常的需求，但是操作没有像后端程序(如:Go)那样方便快捷. 其中由于ArrayBuffer是定长的，所以需要事先分配好长度。不能向Golang那个样可以自动增长，合并两个ArrayBuffer也没有那样方便，通常我们在与服务器通信的时候都需要对二进制数据进行组包。

这里为了解决这些问题，我将对ArrayBuffer,DataView进行分装,提供两个操作二进制的类型，分别实现数据的读取与写入。

#支持高低字节序#

### 数据读取入类ReadBuffer

支持int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64以及二进制数组的写入。

这里还要强调一点，js早期是不支持int64, uint64类型的，所以我们需要通过Bigint来实现。


```js
var ReadBuffer = function(buffer, littleEndian){
    this.pos = 0;
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.littleEndian = (littleEndian == true || littleEndian == "true") ? true : false;
}

ReadBuffer.prototype.readInt8 = function() {
    var v = this.view.getInt8(this.pos);
    this.pos += Int8Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readUint8 = function() {
    var v = this.view.getUint8(this.pos);
    this.pos += Uint8Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readInt16 = function() {
    var v = this.view.getInt16(this.pos, this.littleEndian);
    this.pos += Int16Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readUint16 = function() {
    var v = this.view.getUint16(this.pos, this.littleEndian);
    this.pos += Uint16Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readInt32 = function() {
    var v = this.view.getInt32(this.pos, this.littleEndian);
    this.pos += Int32Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readUint32 = function() {
    var v = this.view.getUint32(this.pos, this.littleEndian);
    this.pos += Uint32Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readInt64 = function() {
    var v = this.view.getBigInt64(this.pos, this.littleEndian);
    this.pos += BigInt64Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readUint64 = function() {
    var v = this.view.getBigUint64(this.pos, this.littleEndian);
    this.pos += BigUint64Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readFloat32 = function() {
    var v = this.view.getFloat32(this.pos, this.littleEndian);
    this.pos += Float32Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readFloat64 = function() {
    var v = this.view.getFloat64(this.pos, this.littleEndian);
    this.pos += Float64Array.BYTES_PER_ELEMENT;
    return v;
}

ReadBuffer.prototype.readString = function() {
    var size = this.readInt16();
    if(size > this.buffer.byteLength) {
        return "";
    }

    var bytes = new Uint8Array(this.readBytes(size));
    var out = [], pos = 0, c = 0;
    while (pos < bytes.byteLength) {
        var c1 = bytes[pos++];
        if (c1 < 128) {
            out[c++] = String.fromCharCode(c1);
        } else if (c1 > 191 && c1 < 224) {
            var c2 = bytes[pos++];
            out[c++] = String.fromCharCode((c1 & 31) << 6 | c2 & 63);
        } else if (c1 > 239 && c1 < 365) {
            // Surrogate Pair
            var c2 = bytes[pos++];
            var c3 = bytes[pos++];
            var c4 = bytes[pos++];
            var u = ((c1 & 7) << 18 | (c2 & 63) << 12 | (c3 & 63) << 6 | c4 & 63) -
                0x10000;
            out[c++] = String.fromCharCode(0xD800 + (u >> 10));
            out[c++] = String.fromCharCode(0xDC00 + (u & 1023));
        } else {
            var c2 = bytes[pos++];
            var c3 = bytes[pos++];
            out[c++] =
                String.fromCharCode((c1 & 15) << 12 | (c2 & 63) << 6 | c3 & 63);
        }
    }
    return out.join('');
}

ReadBuffer.prototype.readBytes = function(size) {
    if(this.pos + size > this.buffer.byteLength) {
        return new ArrayBuffer(0);
    }

    var n = this.buffer.slice(this.pos, this.pos + size);
    this.pos += size;
    return n;
}

ReadBuffer.prototype.bytes = function() {
    return this.buffer.slice(this.pos, this.buffer.byteLength);
}
```

具体用方法：

```js

var buffer = new ReadBuffer(data); // data 为Arraybuffer数据

console.log(buffer.readInt8());
```

这里 bytes 方法返回没有读取完的数据,以方便其它方法进行继续读取


### 数据写入类WriteBuffer

WriteBuffer 类实现与ReadBuffer对应的写功能.

```js
var WriteBuffer = function(buffer, littleEndian) {
    if(buffer == undefined || buffer == null) {
      buffer = new ArrayBuffer(1024);
    }

    this.pos = 0;
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.littleEndian = (littleEndian == true || littleEndian == "true") ? true : false;
  }

  WriteBuffer.prototype.mybeExpand = function(byteLength){
    if(this.pos + byteLength < this.buffer.byteLength) {
      return;
    }

    var newBuffer = new ArrayBuffer(this.buffer.byteLength + byteLength);
    var frame = new Uint8Array(newBuffer);
    frame.set(new Uint8Array(this.buffer), 0)

    this.buffer = newBuffer;
    this.view = new DataView(this.buffer);
  }

  WriteBuffer.prototype.writeInt8 = function(v) {
    this.mybeExpand(Int8Array.BYTES_PER_ELEMENT);
    this.view.setInt8(this.pos, v);
    this.pos += Int8Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeUint8 = function(v) {
    this.mybeExpand(Uint8Array.BYTES_PER_ELEMENT);
    this.view.setUint8(this.pos, v);
    this.pos += Uint8Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeInt16 = function(v) {
    this.mybeExpand(Int16Array.BYTES_PER_ELEMENT);
    this.view.setInt16(this.pos, v, this.littleEndian);
    this.pos += Int16Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeUint16 = function(v) {
    this.mybeExpand(Uint16Array.BYTES_PER_ELEMENT);
    this.view.setUint16(this.pos, v, this.littleEndian);
    this.pos += Uint16Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeInt32 = function(v) {
    this.mybeExpand(Int32Array.BYTES_PER_ELEMENT);
    this.view.setInt32(this.pos, v, this.littleEndian);
    this.pos += Int32Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeUint32 = function(v) {
    this.mybeExpand(Uint32Array.BYTES_PER_ELEMENT);
    this.view.setUint32(this.pos, v, this.littleEndian);
    this.pos += Uint32Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeInt64 = function(v) {
      this.mybeExpand(BigInt64Array.BYTES_PER_ELEMENT);
      this.view.setBigInt64(this.pos, BigInt(v), this.littleEndian);
      this.pos += BigInt64Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeUint64 = function(v) {
    this.mybeExpand(BigUint64Array.BYTES_PER_ELEMENT);
    this.view.setBigUint64(this.pos, BigInt(v), this.littleEndian);
    this.pos += BigUint64Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeFloat32 = function(v) {
    this.mybeExpand(Float32Array.BYTES_PER_ELEMENT);
    this.view.setFloat32(this.pos, v, this.littleEndian);
    this.pos += Float32Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeFloat64 = function(v) {
    this.mybeExpand(Float64Array.BYTES_PER_ELEMENT);
    this.view.setFloat64(this.pos, v, this.littleEndian);
    this.pos += Float64Array.BYTES_PER_ELEMENT;
  }

  WriteBuffer.prototype.writeString = function(str) {
    var out = [], p = 0;
    for (var i = 0; i < str.length; i++) {
      var c = str.charCodeAt(i);
      if (c < 128) {
           out[p++] = c;
       } else if (c < 2048) {
           out[p++] = (c >> 6) | 192;
           out[p++] = (c & 63) | 128;
       } else if (
             ((c & 0xFC00) == 0xD800) && (i + 1) < str.length &&
             ((str.charCodeAt(i + 1) & 0xFC00) == 0xDC00)) {
           // Surrogate Pair
           c = 0x10000 + ((c & 0x03FF) << 10) + (str.charCodeAt(++i) & 0x03FF);
           out[p++] = (c >> 18) | 240;
           out[p++] = ((c >> 12) & 63) | 128;
           out[p++] = ((c >> 6) & 63) | 128;
           out[p++] = (c & 63) | 128;
       } else {
           out[p++] = (c >> 12) | 224;
           out[p++] = ((c >> 6) & 63) | 128;
           out[p++] = (c & 63) | 128;
       }
     }
     this.writeBytes(new Uint8Array(out));
  }

  WriteBuffer.prototype.writeBytes = function(v) {
    this.writeUint16(v.byteLength);
    var newBuffer = new ArrayBuffer(this.buffer.byteLength+v.byteLength);
    var frame = new Uint8Array(newBuffer);

    frame.set(new Uint8Array(this.buffer), 0);
    frame.set(new Uint8Array(v), this.pos);
    this.pos += v.byteLength;
    this.buffer = newBuffer;
    this.view = new DataView(this.buffer);
  }

  WriteBuffer.prototype.bytes = function() {
    return this.buffer.slice(0, this.pos);
  }
```

WriteBuffer 可以传入一个由用户指定长度的buffer,如果没有使用将默认创建一个1024长度buffer。bytes方法负责返回写入的内容，如果内容没有buffer的指定长度那么长。bytes只返回用户写入内容，写多少算多少，不会全部返回buffer.

另外，如果写入数据超出了buffer的长度，那么程序会自动扩容。


### 完整使用方法

```js

var body = new ArrayBuffer(2);
var bodyView = new DataView(body);
bodyView.setInt8(111);
bodyView.setInt8(-111);


var w = new WriteBuffer();
w.writeInt8(1);
w.writeUint32(2);
w.writeInt16(3);
w.writeInt16(4);
w.writeString("中文");
w.writeBytes(body);


var r = new ReadBuffer(w.bytes());
console.log(r.readInt8());
console.log(r.readUInt32());
console.log(r.readInt16());
console.log(r.readInt16());
console.log(r.readString());
console.log(r.bytes());
```
