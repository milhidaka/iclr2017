'use strict';

function run_copy_proc() {
  var vc = new VariableClient($("input[name='server_uri']").val());
  var read_start = Date.now();
  vc.read(Number($("input[name='read_id']").val()), function (err, buf) {
    var read_end = Date.now();
    $("#log").val($("#log").val() + "Read " + buf.byteLength + " bytes in " + (read_end - read_start) + "ms\r\n");
    
    var write_start = Date.now();
    vc.write(buf, Number($("input[name='write_id']").val()), function (err, ret) {
      var write_end = Date.now();
      $("#log").val($("#log").val() + "Wrote " + buf.byteLength + " bytes in " + (write_end - write_start) + "ms\r\n");
    })
  })
  return false;
}

$(function(){
  $("#run_copy").click(run_copy_proc);
});
