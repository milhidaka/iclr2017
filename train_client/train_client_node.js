'use strict';

var fs = require('fs');
var http = require('http');
var url = require('url');
var WebSocketClient = require('websocket').client;
var Sukiyaki = require('milsukiyaki2');
var $M = require('milsushi2');
var variable_client = null;
var control_socket = null;
var control_socket_connection = null;
var config = {};

function write_log(msg) {
  console.log(msg);
}

function start_train() {
  variable_client = new VariableClient(config['variable_server_uri']);

  control_socket = new WebSocketClient();// new WebSocket(config['control_server_uri']);
  control_socket.on('connect', function (connection) {
    write_log('websocket connected');
    control_socket_connection = connection;
    connection.on('error', control_onerror);
    connection.on('close', control_onclose);
    connection.on('message', control_onmessage);
  });
  control_socket.on('connectFailed', function (err) {
    write_log('websocket connection failed: ' + err);
  });
  control_socket.connect(config['control_server_uri'], null);
  return false;
}

var net = null;
var opt = null;
var packer = null;
var use_cl = null;
var shape_info = null;
var batch_division_size = null;
function setup_net_f(callback) {
  use_cl = config['use_cl'];
  if (use_cl) {
    var initcl_ret = $M.initcl();
    write_log("$M.initcl() => " + initcl_ret);
  }
  batch_division_size = config['batch_division_size'];
  http.get(config['netdef_uri'], function (res) {
    var netdef_str = '';
    res.setEncoding('utf8');

    res.on('data', function (chunk) {
      netdef_str += chunk;
    });

    res.on('end', function () {
      var netdef_json = JSON.parse(netdef_str);
      net = new Sukiyaki.Network(netdef_json.net);
      packer = new WeightPack(netdef_json.weight_pack);
      shape_info = netdef_json.shape_info;
      net.init(function () {
        if (use_cl) {
          net.to_cl();
        }
        opt = new Sukiyaki.Optimizers.OptimizerSGD(net, 0.1);
        net.phase = 'train';
        write_log("initializing net finished");
        callback();
      });
    });
  });
  return false;
}


function control_onmessage(e) {
  write_log("Websocket received: " + e.utf8Data);
  var command = JSON.parse(e.utf8Data);
  switch (command["command"]) {
    case 'read_data':
      command_read_data(command);
      break;
    case 'calc_gradient':
      command_calc_gradient(command);
      break;
  }
}

function control_onclose() {
  write_log("Websocket closed");
}

function control_onerror(err) {
  write_log("Websocket error " + err);
}

var data_queue = [];
function command_read_data(command) {
  var vars_to_read = [];
  for (var key in command.vars) {
    vars_to_read.push([key, command.vars[key]])//key, var_id
  }

  var read_data = {};
  var read_next = function () {
    var k_id = vars_to_read.shift();
    var key = k_id[0];
    var var_id = k_id[1];
    variable_client.read(var_id, function (err, buf) {
      read_data[key] = buf;
      if (vars_to_read.length > 0) {
        read_next();
      } else {
        write_log("Completed reading data " + command.vars);
        data_queue.push(read_data);
      }
    });
  }

  read_next();
}

//var current_weight = null;
var current_gradient = null;
var current_gradient_id = null;
function command_calc_gradient(command) {
  var weight_id = command.vars.weight;
  current_gradient_id = command.vars.gradient;
  variable_client.read(weight_id, function (err, buf) {
    console.log('deserializing weight ' + (new Date()));
    packer.unpack(net, buf, false);
    console.log('deserialized weight ' + (new Date()));
    wait_data_calc_gradient();
  });
}

var current_batch_size = null;
var current_batch_division = null;
var current_loss = null;
function wait_data_calc_gradient() {
  if (data_queue.length == 0) {
    write_log("Data is not yet loaded");
    setTimeout(wait_data_calc_gradient, 100);
    return;
  }

  var batch_data = data_queue.shift();
  control_socket_connection.sendUTF(JSON.stringify({ 'command': 'loaded_weight' }));
  write_log("Calculating gradient");
  current_batch_size = batch_data['label'].byteLength / (4);
  current_batch_division = Math.ceil(current_batch_size / batch_division_size);

  var data_sample_size = shape_info.data[0] * shape_info.data[1] * shape_info.data[2];
  opt.zero_grads();
  console.log('forward-backward start ' + (new Date()));
  var batch_div_i = 0;
  var forward_next = function () {
    var batch_div_sample_idx = batch_div_i * batch_division_size;
    var batch_div_count = Math.min(batch_division_size, current_batch_size - batch_div_sample_idx);
    console.log('from ' + batch_div_sample_idx + ' size ' + batch_div_count);
    var data_var = $M.typedarray2mat([shape_info.data[0], shape_info.data[1], shape_info.data[2], batch_div_count],
      'single',
      new Float32Array(batch_data['data'], data_sample_size * 4 * batch_div_sample_idx, data_sample_size * batch_div_count));
    var label_var = $M.typedarray2mat([1, batch_div_count], 'int32',
      new Int32Array(batch_data['label'], 4 * batch_div_sample_idx, batch_div_count));
    if (use_cl) {
      data_var = $M.gpuArray(data_var);
      label_var = $M.gpuArray(label_var);
    }
    net.forward({ data: data_var, label: label_var }, function () {
      net.backward(function () {
        var loss = net.blobs_forward['loss'].get();
        current_loss = loss;
        write_log('Loss: ' + loss);
        batch_div_i++;
        if (batch_div_i == current_batch_division) {
          // end all division
          console.log('forward-backward end ' + (new Date()));
          current_gradient = packer.pack(net, true);//arraybuffer
          opt.release();
          setImmediate(send_gradient, 0);
        } else {
          net.release();
          forward_next();
        }
      });
    });

  }

  forward_next();
}

function send_gradient() {
  write_log("Sending gradient " + current_gradient_id);
  variable_client.write(current_gradient, current_gradient_id, function (err, id) {
    control_socket_connection.sendUTF(JSON.stringify({ 'command': 'stored_gradient', 'gradient_id': current_gradient_id, 'batch_size': current_batch_size, 'gradient_multiplier': current_batch_division, 'loss': current_loss }));
    write_log("Sent gradient message");
  });
}

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
      var request_options = url.parse(uri);
      request_options.method = 'POST';
      var oReq = http.request(request_options, function (res) {
        if (res.statusCode != 200) {
          callback({ msg: 'HTTP status ' + res.statusCode }, null);
        }

        res.on('data', function (chunk) {
          //dummy
        });

        res.on('end', function () {
          if (!id) {
            id = oReq.response.id;
          }
          next_offset += request_size;
          block_index++;
          upload_next();
        });

      });

      oReq.on('error', function (err) {
        callback({ msg: 'HTTP POST Error ' + err }, null);
      })
      oReq.end(new Buffer(subarray));
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
      var request_options = url.parse(uri);
      request_options.method = 'GET';

      var oReq = http.request(request_options, function (res) {
        if (res.statusCode != 200) {
          callback({ msg: 'HTTP status ' + res.statusCode }, null);
        }

        var chunks = [];
        res.on('data', function (chunk) {
          chunks.push(chunk);
        });

        res.on('end', function () {
          // write response to buf
          for (var chunk_i = 0; chunk_i < chunks.length; chunk_i++) {
            var chunk = chunks[chunk_i];
            buf.set(chunk, next_offset);
            next_offset += chunk.length;
          }
          next_block_index++;
          if (next_block_index == total_blocks) {
            // complete
            callback(null, buf.buffer);
          } else {
            download_next();
          }

        });

      });

      oReq.on('error', function (err) {
        callback({ msg: 'HTTP GET Error ' + err }, null);
      })
      oReq.end();
    };

    var url_stat = this.url.stat;

    var get_stat = function () {
      var uri = url_stat + '/' + id;
      http.get(uri, function (res) {
        var body = '';
        res.setEncoding('utf8');

        res.on('data', function (chunk) {
          body += chunk;
        });

        res.on('end', function () {
          var body_obj = JSON.parse(body);
          var file_size = body_obj.size;
          total_blocks = body_obj.blocks.length;
          buf = new Uint8Array(file_size);
          download_next();
        });
      });
    };
    get_stat();

  };

})();

var WeightPack = null;
(function () {
  WeightPack = function (weight_size_data) {
    this.weight_size_data = weight_size_data;
    switch (weight_size_data.format) {
      case 'raw':
        this._pack = pack_raw;
        this._unpack = unpack_raw;
        break;
      case 'eightbit':
        this._pack = Sukiyaki.DettmersWeightCompression.compress_8bit;
        this._unpack = Sukiyaki.DettmersWeightCompression.decompress_8bit;
        break;
    }
  };

  WeightPack.prototype.pack = function (net, gradient) {
    var buf_u8 = new Uint8Array(this.weight_size_data.total_size);
    var buf = buf_u8.buffer;
    for (var i = 0; i < this.weight_size_data.param_sizes.length; i++) {
      var param_meta = this.weight_size_data.param_sizes[i];
      var param_name = gradient ? param_meta.delta_param : param_meta.train_param;
      var weight_mat = net.layer_instances[param_meta.layer][param_name];
      this._pack(weight_mat, buf, param_meta.offset, param_meta.size);
    }

    return buf;
  };

  WeightPack.prototype.unpack = function (net, buf, gradient) {
    for (var i = 0; i < this.weight_size_data.param_sizes.length; i++) {
      var param_meta = this.weight_size_data.param_sizes[i];
      var param_name = gradient ? param_meta.delta_param : param_meta.train_param;
      var weight_mat = net.layer_instances[param_meta.layer][param_name];
      this._unpack(weight_mat, buf, param_meta.offset, param_meta.size);
    }
  };

  var pack_raw = function (weight_mat, dst_buf, dst_offset, dst_size) {
    var buf_view = new Float32Array(dst_buf, dst_offset, dst_size / 4);
    // var wvec = $M.reshape(weight_mat, [-1, 1]);
    // var stdvar = $M.sum(wvec);
    // console.log("sum: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + dst_offset + " dst_size " + dst_size);
    // wvec.destruct();
    // stdvar.destruct();
    weight_mat.getdatacopy(null, null, buf_view);
  };

  var unpack_raw = function (weight_mat, src_buf, src_offset, src_size) {
    var buf_view = new Float32Array(src_buf, src_offset, src_size / 4);
    weight_mat.setdata(buf_view);
    // var wvec = $M.reshape(weight_mat, [-1, 1]);
    // var stdvar = $M.std(wvec);
    // console.log("unpack std: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + src_offset + " dst_size " + src_size);
    // wvec.destruct();
    // stdvar.destruct();
  };

  var naive_pack_eightbit_kernel = null;
  var pack_eightbit = function (weight_mat, dst_buf, dst_offset, dst_size) {
    var buf_view = new Uint8Array(dst_buf, dst_offset, dst_size - 4);
    if (!naive_pack_eightbit_kernel) {
      naive_pack_eightbit_kernel = $M.CL.createKernel([
        '__kernel void kernel_func(__global uchar *weight_packed, __global float *weight_raw, uint length)',
        '{',
        'uint i = get_global_id(0);',
        'if (i >= length) {return;}',
        'float val = weight_raw[i];',
        'uchar signbit = val < 0.0F ? 128 : 0;',
        'float logval = log2(fabs(val));',
        'if (logval < -64.0F) {logval = -64.0F;}',
        'else if (logval > 63.0F) {logval = 63.0F;}',
        'uchar exponent = (uchar)(logval + 64.0F);',
        'uchar packed = signbit + exponent;',
        'weight_packed[i] = packed;',
        '}'].join('\n'));
    }

    var WebCL = $M.CL.WebCL;
    var weight_packed = new $M.CL.MatrixCL($M.sizejsa(weight_mat), 'uint8');
    var numel = $M.numel(weight_packed);
    $M.CL.executeKernel(naive_pack_eightbit_kernel, [
      { access: WebCL.MEM_WRITE_ONLY, datum: weight_packed },
      { access: WebCL.MEM_READ_ONLY, datum: weight_mat },
      { datum: numel, type: WebCL.type.UINT }
    ], numel);

    weight_packed.getdatacopy(null, null, buf_view);
    weight_packed.destruct();
  };

  var naive_unpack_eightbit_kernel = null;
  var unpack_eightbit = function (weight_mat, src_buf, src_offset, src_size) {
    var buf_view = new Uint8Array(src_buf, src_offset, src_size - 4);
    if (!naive_unpack_eightbit_kernel) {
      naive_unpack_eightbit_kernel = $M.CL.createKernel([
        '__kernel void kernel_func(__global float *weight_raw, __global uchar *weight_packed, uint length)',
        '{',
        'uint i = get_global_id(0);',
        'if (i >= length) {return;}',
        'uchar packed = weight_packed[i];',
        'float expval = exp2((float)(packed & 0x7F) - 64.0F);',
        'float val = packed & 0x80 ? -expval : expval;',
        'weight_raw[i] = val;',
        '}'].join('\n'));
    }

    var WebCL = $M.CL.WebCL;
    var weight_packed = new $M.CL.MatrixCL($M.sizejsa(weight_mat), 'uint8');
    weight_packed.setdata(buf_view);
    var numel = $M.numel(weight_packed);
    $M.CL.executeKernel(naive_unpack_eightbit_kernel, [
      { access: WebCL.MEM_WRITE_ONLY, datum: weight_mat },
      { access: WebCL.MEM_READ_ONLY, datum: weight_packed },
      { datum: numel, type: WebCL.type.UINT }
    ], numel);

    weight_packed.destruct();
  };
})();

function main() {
  config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  setup_net_f(function () {
    start_train();
  });
}

main();
