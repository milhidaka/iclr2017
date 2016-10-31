'use strict';
var fs = require('fs');
var convnetjs = require('./convnet-min');

var dataset = null;
var batch_size = 64;
function load_dataset() {
  dataset = {};
  load_dataset_obj('train-labels-idx1-ubyte', 'train_labels', parse_dataset_labels);
  load_dataset_obj('t10k-labels-idx1-ubyte', 'test_labels', parse_dataset_labels);
  load_dataset_obj('train-images-idx3-ubyte', 'train_images', parse_dataset_images);
  load_dataset_obj('t10k-images-idx3-ubyte', 'test_images', parse_dataset_images);
}

function load_dataset_obj(url, key, parser) {
  dataset[key] = parser(fs.readFileSync(url).buffer);
  console.log('dataset ' + key + ' loaded');
}

function parse_dataset_images(buf) {
  var rawarray = new Uint8Array(buf);
  var image_size = 28 * 28;
  var n_samples = (rawarray.length - 16) / image_size;
  var images = [];
  for (var i = 0; i < n_samples; i++) {
    var image_jsarray = [];
    for (var j = 0; j < image_size; j++) {
      image_jsarray.push(rawarray[16 + i * image_size + j] / 256);
    }

    var vol = new convnetjs.Vol();
    vol.fromJSON({ "sx": 28, "sy": 28, "depth": 1, "w": image_jsarray });
    images.push(vol);
  }

  return images;
}

function parse_dataset_labels(buf) {
  var rawarray = new Uint8Array(buf);
  var labels = [];
  var n_samples = rawarray.length - 8;
  for (var i = 0; i < n_samples; i++) {
    labels.push(rawarray[8 + i]);
  }
  return labels;
}

function make_net_trainer() {
  var layer_defs = [];
  layer_defs.push({ type: 'input', out_sx: 28, out_sy: 28, out_depth: 1 });
  layer_defs.push({ type: 'conv', sx: 5, filters: 20, stride: 1, pad: 0 });
  layer_defs.push({ type: 'pool', sx: 2, stride: 2 });
  layer_defs.push({ type: 'conv', sx: 5, filters: 50, stride: 1, pad: 0 });
  layer_defs.push({ type: 'pool', sx: 2, stride: 2 });
  layer_defs.push({ type: 'fc', num_neurons: 500, activation: 'relu' });
  layer_defs.push({ type: 'softmax', num_classes: 10 });
  var net = new convnetjs.Net();
  net.makeLayers(layer_defs);

  var trainer = new convnetjs.SGDTrainer(net, {method:'sgd',
                                            learning_rate: 0.01, 
                                            l2_decay: 0.0,
                                            momentum: 0.9,
                                            batch_size: batch_size});
  return {net: net, trainer: trainer};
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
  console.log('' + iter_time_begin + ' iteration '+iter_idx);
  for (var i = 0; i < batch_size; i++) {
    train_status = trainer.train(images[data_idx], labels[data_idx]);
    data_idx = (data_idx + 1) % labels.length;
  }

  var iter_time_end = new Date();
  var iter_ms = iter_time_end - iter_time_begin;
  total_train_ms += iter_ms;
  console.log('elapsed ' + iter_ms + 'ms loss: ' + train_status.loss);
  iter_idx++;
  if (iter_idx % 10 == 0) {
    console.log('average speed = ' + (iter_idx * batch_size / (total_train_ms / 1000) + ' images/sec'));
  }
  if ((iter_idx + 1) * batch_size < labels.length) {
    setTimeout(process_one_iter, 0);
  } else {
    console.log('end of benchmark');
  }
}

function bench_node() {
  var net_trainer = make_net_trainer();
  net = net_trainer.net;
  trainer = net_trainer.trainer;
  process_one_iter();
}

function main() {
  load_dataset();
  bench_node();
}

main();
