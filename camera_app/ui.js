'use strict';

function write_msg(m) {
  var msg_box = $("#msg");
  msg_box.val(msg_box.val() + "\n" + m);
  msg_box.scrollTop(msg_box[0].scrollHeight - msg_box.height());
}

function start_camera(callback) {
  write_msg('Initializing video input device');
  navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || window.navigator.mozGetUserMedia;
  var wURL = window.URL || window.webkitURL;
  var video = $("#camera-in")[0];
  navigator.getUserMedia({ video: { facingMode: 'environment' }, audio: false },
    function (stream) {
      video.src = wURL.createObjectURL(stream);
      write_msg('Video initialization ok');
      callback(true);
    },
    function (err) {
      write_msg('Video initialization failed: ' + err.name);
      callback(false);
    });
}

$(function () {
  $("#start-camera").click(function(){start_camera(function(){})});

  $("#video-capture").click(function () {
    console.log('capture');
    var video = $("#camera-in")[0];
    var canvas = $("#recognize-in")[0];
    var context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, 28, 28);
    return false;
  });

  $("#load-net").click(function () {
    load_network_net("lenet_core.json?d=" + (Date.now()), "lenet_invert_1000?d=" + (Date.now()), function(){});
    return false;
  });

  $("#recognize").click(function () {
    do_recognize($("#recognize-in")[0]);
    return false;
  })

  $("#autorecognize").click(function () {
    setInterval(function () {
      var video = $("#camera-in")[0];
      var canvas = $("#recognize-in")[0];
      var context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, 28, 28);
      do_recognize(context);
    }, 500);
    return false;
  })
});
