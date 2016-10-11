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
function setup_net_f() {
  use_cl = Boolean($("input[name='use_cl']").prop('checked'));
  if (use_cl) {
    var initcl_ret = $M.initcl();
    write_log("$M.initcl() => " + initcl_ret);
  }
  $.getJSON($("input[name='netdef_uri']").val(), function (netdef_json) {
    net = new Sukiyaki.Network(netdef_json.net);
    packer = new WeightPack(netdef_json.weight_pack);
    shape_info = netdef_json.shape_info;
    net.init(function(){
      if (use_cl) {
        net.to_cl();
      }
      opt = new Sukiyaki.Optimizers.OptimizerSGD(net, 0.1);
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
function wait_data_calc_gradient() {
  if (data_queue.length == 0) {
    write_log("Data is not yet loaded");
    setTimeout(wait_data_calc_gradient, 100);
    return;
  }

  var batch_data = data_queue.shift();
  write_log("Calculating gradient");
  current_batch_size = batch_data['label'].byteLength / (4);

  var data_var = $M.typedarray2mat([shape_info.data[0], shape_info.data[1], shape_info.data[2],current_batch_size],
  'single', new Float32Array(batch_data['data']));
  var dvvec = $M.reshape(data_var, [-1 , 1]);
  console.log("data sum: " + $M.sum(dvvec).get());
  var label_var = $M.typedarray2mat([1, current_batch_size], 'int32', new Int32Array(batch_data['label']));
  if (use_cl) {
    data_var = $M.gpuArray(data_var);
    label_var = $M.gpuArray(label_var);
  }
  console.log('forward-backward start ' + (new Date()));
  opt.zero_grads();
  net.forward({data:data_var, label:label_var}, function () {
    net.backward(function () {
      var loss = net.blobs_forward['loss'].get();
      write_log('Loss: ' + loss);
      console.log('forward-backward end ' + (new Date()));
      current_gradient = packer.pack(net, true);//arraybuffer
      console.log('gradient size: ' + current_gradient.byteLength);
      opt.release();
      setImmediate(send_gradient, 0);
    });
  });
}

function send_gradient() {
  write_log("Sending gradient " + current_gradient_id);
  variable_client.write(current_gradient, current_gradient_id, function (err, id) {
    control_socket.send(JSON.stringify({ 'command': 'stored_gradient', 'gradient_id': current_gradient_id, 'batch_size':current_batch_size, 'gradient_multiplier': 1 }));
    write_log("Sent gradient message");
  });
}
