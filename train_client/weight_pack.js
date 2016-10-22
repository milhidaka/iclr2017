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
      case 'eightbit':
        this._pack = pack_eightbit;
        this._unpack = unpack_eightbit;
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
    var buf_view = new Float32Array(dst_buf, dst_offset, dst_size / 4);
    // var wvec = $M.reshape(weight_mat, [-1, 1]);
    // var stdvar = $M.sum(wvec);
    // console.log("sum: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + dst_offset + " dst_size " + dst_size);
    // wvec.destruct();
    // stdvar.destruct();
    weight_mat.getdatacopy(null, null, buf_view);
  };

  var unpack_raw = function (weight_mat, src_buf, src_offset, src_size) {
    var buf_view = new Float32Array(src_buf, src_offset, src_size / 4);
    weight_mat.setdata(buf_view);
    // var wvec = $M.reshape(weight_mat, [-1, 1]);
    // var stdvar = $M.std(wvec);
    // console.log("unpack std: " + stdvar.get() + " head: " + weight_mat.get() + " size: " + $M.sizejsa(weight_mat) + " sumsize: " + $M.sizejsa(stdvar) + " dst_offset " + src_offset + " dst_size " + src_size);
    // wvec.destruct();
    // stdvar.destruct();
  };

  var naive_pack_eightbit_kernel = null;
  var pack_eightbit = function (weight_mat, dst_buf, dst_offset, dst_size) {
    var buf_view = new Uint8Array(dst_buf, dst_offset, dst_size - 4);
    if (!naive_pack_eightbit_kernel) {
      naive_pack_eightbit_kernel = $M.CL.createKernel([
        '__kernel void kernel_func(__global uchar *weight_packed, __global float *weight_raw, uint length)',
        '{',
        'uint i = get_global_id(0);',
        'if (i >= length) {return;}',
        'float val = weight_raw[i];',
        'uchar signbit = val < 0.0F ? 128 : 0;',
        'float logval = log2(fabs(val));',
        'if (logval < -64.0F) {logval = -64.0F;}',
        'else if (logval > 63.0F) {logval = 63.0F;}',
        'uchar exponent = (uchar)(logval + 64.0F);',
        'uchar packed = signbit + exponent;',
        'weight_packed[i] = packed;',
        '}'].join('\n'));
    }

    var WebCL = $M.CL.WebCL;
    var weight_packed = new $M.CL.MatrixCL($M.sizejsa(weight_mat), 'uint8');
    var numel = $M.numel(weight_packed);
    $M.CL.executeKernel(naive_pack_eightbit_kernel, [
      { access: WebCL.MEM_WRITE_ONLY, datum: weight_packed },
      { access: WebCL.MEM_READ_ONLY, datum: weight_mat },
      { datum: numel, type: WebCL.type.UINT }
    ], numel);

    weight_packed.getdatacopy(null, null, buf_view);
    weight_packed.destruct();
  };

  var naive_unpack_eightbit_kernel = null;
  var unpack_eightbit = function (weight_mat, src_buf, src_offset, src_size) {
    var buf_view = new Uint8Array(src_buf, src_offset, src_size - 4);
    if (!naive_unpack_eightbit_kernel) {
      naive_unpack_eightbit_kernel = $M.CL.createKernel([
        '__kernel void kernel_func(__global float *weight_raw, __global uchar *weight_packed, uint length)',
        '{',
        'uint i = get_global_id(0);',
        'if (i >= length) {return;}',
        'uchar packed = weight_packed[i];',
        'float expval = exp2((float)(packed & 0x7F) - 64.0F);',
        'float val = packed & 0x80 ? -expval : expval;',
        'weight_raw[i] = val;',
        '}'].join('\n'));
    }

    var WebCL = $M.CL.WebCL;
    var weight_packed = new $M.CL.MatrixCL($M.sizejsa(weight_mat), 'uint8');
    weight_packed.setdata(buf_view);
    var numel = $M.numel(weight_packed);
    $M.CL.executeKernel(naive_unpack_eightbit_kernel, [
      { access: WebCL.MEM_WRITE_ONLY, datum: weight_mat },
      { access: WebCL.MEM_READ_ONLY, datum: weight_packed },
      { datum: numel, type: WebCL.type.UINT }
    ], numel);

    weight_packed.destruct();
  };
})();
