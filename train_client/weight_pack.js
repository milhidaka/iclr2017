'use strict';

var WeightPack = null;
(function () {
  WeightPack = function (weight_size_data) {
    this.weight_size_data = weight_size_data;
    switch (weight_size_data.format) {
      case 'raw':
        this._pack = pack_raw;
        this._unpack = unpack_raw;
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
    // buffer with offset seems to be ignored in $M.getdatacopy()?
    // workaround with tmp buffer
    var buf_view = new Float32Array(dst_buf, dst_offset, dst_size / 4);
    var wvec = $M.reshape(weight_mat, [-1, 1]);
    var stdvar = $M.sum(wvec);
    console.log("sum: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + dst_offset + " dst_size " + dst_size);
    wvec.destruct();
    stdvar.destruct();
    var tmp_buf = new Float32Array(dst_size / 4);
    weight_mat.getdatacopy(null, null, tmp_buf);
    buf_view.set(tmp_buf);
  };

  var unpack_raw = function (weight_mat, src_buf, src_offset, src_size) {
    // same workaround with pack_raw
    var buf_view = new Float32Array(src_buf, src_offset, src_size / 4);
    var tmp_buf = new Float32Array(src_size / 4);
    tmp_buf.set(buf_view);
    weight_mat.setdata(tmp_buf);
    var wvec = $M.reshape(weight_mat, [-1, 1]);
    var stdvar = $M.std(wvec);
    console.log("unpack std: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + src_offset + " dst_size " + src_size);
    wvec.destruct();
    stdvar.destruct();
  };
})();
