'use strict';

var $M = milsushi2;
var Sukiyaki = milsukiyaki2;
function load_network_net(netdef_url, weight_url, callback) {
  write_msg('Loading network');
  $.getJSON(netdef_url, function (netdef_json) {
    var oReq = new XMLHttpRequest();
    oReq.open("GET", weight_url, true);
    oReq.responseType = "arraybuffer";

    oReq.onload = function (oEvent) {
      var arrayBuffer = oReq.response; // Note: not oReq.responseText
      load_network_local(netdef_json, arrayBuffer, callback);
    };

    oReq.send(null);
  }
  );
}

var net = null;
function load_network_local(netdef_json, weight_arraybuf, callback) {
  net = new Sukiyaki.Network(netdef_json);
  net.init(function () {
    Sukiyaki.ArraySerializer.load(new Uint8Array(weight_arraybuf), net);
    write_msg('Network loaded');
    callback();
  });
}

var now_recognizing = false;
function do_recognize(canvas_context) {
  if (!now_recognizing) {
    now_recognizing = true;
    var imagedata = canvas_context.getImageData(0, 0, 28, 28);// get pixel data from canvas
    var image = $M.typedarray2mat([4, 28, 28], 'uint8', new Uint8Array(imagedata.data));// channel, width, height (in fortran-order)
    image = image.get(1, $M.colon(), $M.colon());// extract single color channel
    image = $M.permute(image, [3, 2, 1]);// transpose to height, width, channel
    net.forward({ 'data': image }, function () {// forward propagation
      var pred = net.blobs_forward['pred'];// prediction layer output
      var max_index = $M.argmax(pred).I.get();// get matrix index of highest score (1-origin)
      var predicted_number = max_index - 1;
      document.getElementById('result').textContent = predicted_number.toString();
      net.release();
      now_recognizing = false;
    });
  }
}
