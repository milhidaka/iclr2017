'use strict';

var $M = milsushi2;
var Sukiyaki = milsukiyaki2;

$(function () {

});

var dataset = null;
var batch_size = 64;
var use_cl = null;
function load_dataset() {
  dataset = {};
  load_dataset_obj('train-labels-idx1-ubyte', 'train_labels', parse_dataset_labels);
  load_dataset_obj('t10k-labels-idx1-ubyte', 'test_labels', parse_dataset_labels);
  load_dataset_obj('train-images-idx3-ubyte', 'train_images', parse_dataset_images);
  load_dataset_obj('t10k-images-idx3-ubyte', 'test_images', parse_dataset_images);
}

function load_dataset_obj(url, key, parser) {
  var oReq = new XMLHttpRequest();
  oReq.open('GET', url, true);
  oReq.responseType = 'arraybuffer';
  oReq.onload = function (oEvent) {
    if (oReq.status != 200) {
      // error
      console.log('dataset load failed');
    } else {
      // write response to buf
      dataset[key] = parser(oReq.response);
      $("#bench_result").text($("#bench_result").text() + ',' + key);
      console.log('dataset ' + key + ' loaded');
    }

  }

  oReq.send(null);
}

function parse_dataset_images(buf) {
  var rawarray = new Uint8Array(buf, 16);
  var float32array = new Float32Array(rawarray.length);
  float32array.set(rawarray);
  var images = $M.typedarray2mat([28, 28, 1, rawarray.length / (28 * 28 * 1)], 'single', float32array);
  images = $M.times(images, 1 / 256);

  return images;
}

function parse_dataset_labels(buf) {
  var rawarray = new Uint8Array(buf, 8);
  var int32array = new Int32Array(rawarray.length);
  int32array.set(rawarray);
  var labels = $M.typedarray2mat([1, rawarray.length], 'int32', int32array);
  return labels;
}

function make_net_trainer(callback) {
  var layer_defs = [
    { name: "conv1", type: "convolution_2d", params: { in_size: 1, out_size: 20, ksize: 5, stride: 1, pad: 0 }, inputs: ["data"], outputs: ["conv1"] },
    { name: "pool1", type: "pooling_2d", params: { type: "max", ksize: 2, stride: 2, pad: 0 }, inputs: ["conv1"], outputs: ["pool1"] },
    { name: "conv2", type: "convolution_2d", params: { in_size: 20, out_size: 50, ksize: 5, stride: 1, pad: 0 }, inputs: ["pool1"], outputs: ["conv2"] },
    { name: "pool2", type: "pooling_2d", params: { type: "max", ksize: 2, stride: 2, pad: 0 }, inputs: ["conv2"], outputs: ["pool2"] },
    { name: "fc3", type: "linear", params: { in_shape: [4, 4, 50], out_size: 500 }, inputs: ["pool2"], outputs: ["fc3"] },
    { name: "relu3", type: "relu", params: {}, inputs: ["fc3"], outputs: ["relu3"] },
    { name: "fc4", type: "linear", params: { in_shape: [500], out_size: 10 }, inputs: ["relu3"], outputs: ["pred"] },
    { name: "l", type: "softmax_cross_entropy", params: {}, inputs: ["pred", "label"], outputs: ["loss"] }];
  var net = new Sukiyaki.Network(layer_defs);
  net.init(function () {
    if (use_cl) {
      net.to_cl();
    }
    var trainer = new Sukiyaki.Optimizers.OptimizerMomentumSGD(net, 1e-2, 0.9);
    net.phase = "train";
    callback({ net: net, trainer: trainer });
  });
}

var net = null;
var trainer = null;
var data_idx = 0;
var iter_idx = 0;
var total_train_ms = 0;
var train_status = null;
function process_one_iter() {
  var images = dataset['train_images'];
  var labels = dataset['train_labels'];
  var iter_time_begin = new Date();
  console.log('' + iter_time_begin + ' iteration ' + iter_idx);
  var n_samples = $M.size(images, 4);
  var slice_low = iter_idx * batch_size + 1;
  var slice_high = (iter_idx + 1) * batch_size;
  var batch_images = images.get($M.colon(), $M.colon(), $M.colon(), $M.colon(slice_low, slice_high));
  var batch_labels = labels.get($M.colon(), $M.colon(slice_low, slice_high));
  var input_vars = { 'data': batch_images, 'label': batch_labels };
  var iter_time_beforeupdate = new Date();
  trainer.update(input_vars, function () {
    var loss = net.blobs_forward['loss'].get();
    trainer.release();
    var iter_time_end = new Date();
    var iter_ms = iter_time_end - iter_time_begin;
    total_train_ms += iter_ms;
    console.log('elapsed ' + iter_ms + 'ms loss: ' + loss);
    iter_idx++;
    if (iter_idx % 10 == 0) {
      console.log('average speed = ' + (iter_idx * batch_size / (total_train_ms / 1000) + ' images/sec'));
      $("#bench_result").text('average speed = ' + (iter_idx * batch_size / (total_train_ms / 1000)) + ' images/sec' + ' loss = ' + loss);
    }
    if (slice_high + batch_size < n_samples) {
      setImmediate(process_one_iter);
    } else {
      console.log('end of benchmark');
    }
  });
}

function bench_browser() {
  use_cl = Boolean($("input[name='use_cl']").prop('checked'));
  if (use_cl) {
    var initcl_ret = $M.initcl();
    $("#bench_result").text('WebCL initialize: ' + initcl_ret);
    dataset['train_images'] = $M.gpuArray(dataset['train_images']);
    dataset['train_labels'] = $M.gpuArray(dataset['train_labels']);
  }
  make_net_trainer(function (net_trainer) {
    net = net_trainer.net;
    trainer = net_trainer.trainer;
    process_one_iter();
  });
}
