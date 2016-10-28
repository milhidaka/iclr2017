'use strict';

var $M = milsushi2;
var Sukiyaki = milsukiyaki2;
var variable_client = null;
var control_socket = null;
$(function () {
  $("#run_training").click(start_train);
  $("#setup_net").click(setup_net_f);
});

function write_log(msg) {
  $("#log").val($("#log").val() + msg + "\r\n");
}

function start_train() {
  variable_client = new VariableClient($("input[name='variable_server_uri']").val());
  control_socket = new WebSocket($("input[name='control_server_uri']").val());
  control_socket.onopen = control_onopen;
  control_socket.onmessage = control_onmessage;
  control_socket.onerror = control_onerror;
  control_socket.onclose = control_onclose;
  return false;
}

var net = null;
var opt = null;
var packer = null;
var use_cl = null;
var shape_info = null;
var batch_division_size = null;
function setup_net_f() {
  use_cl = Boolean($("input[name='use_cl']").prop('checked'));
  if (use_cl) {
    var initcl_ret = $M.initcl();
    write_log("$M.initcl() => " + initcl_ret);
  }
  batch_division_size = Number($("input[name='batch_division_size']").val());
  $.getJSON($("input[name='netdef_uri']").val(), function (netdef_json) {
    net = new Sukiyaki.Network(netdef_json.net);
    packer = new WeightPack(netdef_json.weight_pack);
    shape_info = netdef_json.shape_info;
    net.init(function () {
      if (use_cl) {
        net.to_cl();
      }
      opt = new Sukiyaki.Optimizers.OptimizerMomentumSGD(net, 1e-3);
      net.phase = 'train';
      write_log("initializing net finished");
    });
  });
  return false;
}

function control_onopen() {
  write_log("Websocket opened");
}

function control_onmessage(e) {
  write_log("Websocket received: " + e.data);
  var command = JSON.parse(e.data);
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
        write_log("Completed reading data " + JSON.stringify(command.vars));
        data_queue.push(read_data);
      }
    });
  }

  read_next();
}

//var current_weight = null;
var current_gradient = null;
var current_gradient_id = null;
var remaining_iters = 0;
var lr = 0;
function command_calc_gradient(command) {
  var weight_id = command.vars.weight;
  current_gradient_id = command.vars.gradient;
  remaining_iters = command.iterations;
  lr = command.lr;
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
  control_socket.send(JSON.stringify({ 'command': 'loaded_weight' }));
  write_log("Calculating gradient");
  current_batch_size = batch_data['label'].byteLength / (4);
  current_batch_division = Math.ceil(current_batch_size / batch_division_size);

  var data_sample_size = shape_info.data[0] * shape_info.data[1] * shape_info.data[2];
  opt.lr = lr / current_batch_division;//gradient is added in each division, so reduce learning rate to offset
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
          opt.do_update();
          opt.release();
          console.log('forward-backward end ' + (new Date()));
          remaining_iters--;
          if (remaining_iters <= 0) {
            current_gradient = packer.pack(net);
            send_gradient();
          } else {
            control_socket.send(JSON.stringify({ 'command': 'iteration_finished', 'loss': current_loss, 'remaining_iters': remaining_iters }));
            wait_data_calc_gradient();
          }
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
    control_socket.send(JSON.stringify({ 'command': 'stored_gradient', 'gradient_id': current_gradient_id, 'loss': current_loss }));
    write_log("Sent gradient message");
  });
}
