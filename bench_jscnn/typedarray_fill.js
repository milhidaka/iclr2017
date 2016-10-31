(function(){
  // polyfill for firefox < 37
  // taken from https://developer.mozilla.org/ja/docs/Web/JavaScript/Reference/Global_Objects/Array/fill
  var fill = function(value) {
    // Steps 1-2.
    if (this == null) {
      throw new TypeError('this is null or not defined');
    }

    var O = Object(this);

    // Steps 3-5.
    var len = O.length >>> 0;

    // Steps 6-7.
    var start = arguments[1];
    var relativeStart = start >> 0;

    // Step 8.
    var k = relativeStart < 0 ?
      Math.max(len + relativeStart, 0) :
      Math.min(relativeStart, len);

    // Steps 9-10.
    var end = arguments[2];
    var relativeEnd = end === undefined ?
      len : end >> 0;

    // Step 11.
    var final = relativeEnd < 0 ?
      Math.max(len + relativeEnd, 0) :
      Math.min(relativeEnd, len);

    // Step 12.
    while (k < final) {
      O[k] = value;
      k++;
    }

    // Step 13.
    return O;
  };

  if (!Uint8Array.prototype.fill) {
    Uint8Array.prototype.fill = fill;
    Uint8ClampedArray.prototype.fill = fill;
    Uint16Array.prototype.fill = fill;
    Uint32Array.prototype.fill = fill;
    Int8Array.prototype.fill = fill;
    Int16Array.prototype.fill = fill;
    Int32Array.prototype.fill = fill;
    Float32Array.prototype.fill = fill;
    Float64Array.prototype.fill = fill;
  }
})();
