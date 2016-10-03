'use strict';

var variable_client = null;
var control_socket = null;
$(function () {
  $("#run_training").click(start_train);
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

var current_weight = null;
var current_gradient = null;
var current_gradient_id = null;
function command_calc_gradient(command) {
  var weight_id = command.vars.weight;
  current_gradient_id = command.vars.gradient;
  variable_client.read(weight_id, function (err, buf) {
    current_weight = buf;
    wait_data_calc_gradient();
  });
}

function wait_data_calc_gradient() {
  if (data_queue.length == 0) {
    write_log("Data is not yet loaded");
    setTimeout(wait_data_calc_gradient, 100);
    return;
  }

  var batch_data = data_queue.shift();
  write_log("Calculating gradient (dummy)");

  current_gradient = current_weight;
  setTimeout(send_gradient, 0);
}

function send_gradient() {
  write_log("Sending gradient " + current_gradient_id);
  variable_client.write(current_gradient, current_gradient_id, function (err, id) {
    control_socket.send(JSON.stringify({ 'command': 'stored_gradient' }));
    write_log("Sent gradient message");
  });
}
