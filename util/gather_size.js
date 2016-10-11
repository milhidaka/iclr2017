var $M = require('milsushi2');
var Sukiyaki = require('milsukiyaki2');
var fs = require('fs');

function generate_network(path, callback) {
  var layers = JSON.parse(fs.readFileSync(path, 'utf8'));
  var net = new Sukiyaki.Network(layers);
  net.init(function () {
    callback(net);
  });
}

function gather_param_dims(net) {
  var results = [];
  for (var layer_name in net.layer_instances) {
    if (net.layer_instances.hasOwnProperty(layer_name)) {
      var layer_inst = net.layer_instances[layer_name];
      if (!layer_inst.train_params) {
        continue;
      }
      var params_names = layer_inst.train_params;
      var delta_params_names = layer_inst.delta_params;
      for (var i = 0; i < params_names.length; i++) {
        var train_param_name = params_names[i];
        var delta_param_name = delta_params_names[i];
        var weight = layer_inst[train_param_name];
        var weight_dim = $M.numel(weight);//number of dims (not byte)
        results.push({ layer: layer_name, train_param: train_param_name, delta_param: delta_param_name, dims: weight_dim });
      }
    }
  }
  return results;
}

function get_format_size_func(format) {
  switch (format) {
    case 'raw':
      return function (dims) { return dims * 4; };
    case 'eightbit':
      return function (dims) { return dims + 4; };
    default:
      throw new Error('Unknown format');
  }
}

function make_size_data(param_dims, format) {
  var size_calc_f = get_format_size_func(format);
  var param_sizes = [];
  var offset = 0;
  for (var i = 0; i < param_dims.length; i++) {
    var param_dim = param_dims[i];
    var size = size_calc_f(param_dim.dims);
    param_sizes.push({ layer: param_dim.layer, train_param: param_dim.train_param, delta_param: param_dim.delta_param, size: size, offset: offset });
    offset += size;
  }

  return { param_sizes: param_sizes, total_size: offset, format: format }
}

function main() {
  var netdef_path = process.argv[2];
  var dst_path = process.argv[3];
  var format = process.argv[4];
  generate_network(netdef_path, function (net) {
    var param_dims = gather_param_dims(net);
    var size_data = make_size_data(param_dims, format);
    fs.writeFileSync(dst_path, JSON.stringify(size_data));
  });
}

main();
