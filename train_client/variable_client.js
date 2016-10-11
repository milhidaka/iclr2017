'use strict';

var VariableClient = null;
(function () {
  VariableClient = function (url_base, split_size) {
    if (!split_size) {
      split_size = 1024 * 1024 * 100;
    }
    this.split_size = split_size;
    this.url_base = url_base;
    this.url = {
      'write': url_base + '/write',
      'read': url_base + '/read',
      'stat': url_base + '/stat'
    };
  };

  VariableClient.prototype.write = function (buf, id, callback) {
    // write arraybuffer to server
    // callback(err, {id:id})
    var next_offset = 0;
    var split_size = this.split_size;
    var url_write = this.url.write;
    var block_index = 0;

    var upload_next = function () {
      var remaining_size = buf.byteLength - next_offset;
      var request_size = Math.min(remaining_size, split_size);
      if (request_size == 0) {
        //upload complete
        callback(null, { id: id });
        return;
      }
      var close = (request_size == remaining_size) ? 1 : 0;
      var uri = url_write + '/' + id + '?close=' + close;
      var subarray = new Uint8Array(buf, next_offset, request_size);
      var oReq = new XMLHttpRequest();
      oReq.open('POST', uri, true);
      oReq.setRequestHeader("Content-Type", "application/octet-stream");
      oReq.responseType = 'json';

      oReq.onload = function (oEvent) {
        if (oReq.status != 200) {
          // error
          callback({ msg: 'HTTP ' + oReq.status }, null);
        } else {
          if (!id) {
            id = oReq.response.id;
          }
          next_offset += request_size;
          block_index++;
          upload_next();
        }
      };

      oReq.send(subarray);
    }

    upload_next();
  };

  VariableClient.prototype.read = function (id, callback) {
    // read arraybuffer from server
    // callback(err, arraybuffer)
    var buf = null;
    var total_blocks = null;
    var next_block_index = 0;
    var next_offset = 0;
    var url_read = this.url.read;
    var download_next = function () {
      var uri = url_read + '/' + id + '?block_index=' + next_block_index;
      var oReq = new XMLHttpRequest();
      oReq.open('GET', uri, true);
      oReq.responseType = 'arraybuffer';

      oReq.onload = function (oEvent) {
        if (oReq.status != 200) {
          // error
          callback({ msg: 'HTTP ' + oReq.status }, null);
        } else {
          // write response to buf
          var resbuf = new Uint8Array(oReq.response);
          buf.set(resbuf, next_offset);
          next_block_index++;
          next_offset += resbuf.length;
          if (next_block_index == total_blocks) {
            // complete
            callback(null, buf.buffer);
          } else {
            download_next();
          }
        }

      }

      oReq.send(null);

    };

    var url_stat = this.url.stat;
    var get_stat = function () {
      var uri = url_stat + '/' + id;
      var oReq = new XMLHttpRequest();
      oReq.open('GET', uri, true);
      oReq.responseType = 'json';
      oReq.onload = function (oEvent) {
        if (oReq.status != 200) {
          // error
          callback({ msg: 'HTTP ' + oReq.status }, null);
        } else {
          var file_size = oReq.response.size;
          total_blocks = oReq.response.blocks.length;
          buf = new Uint8Array(file_size);
          download_next();
        }
      };

      oReq.send(null);
    };
    get_stat();

  };

})();
