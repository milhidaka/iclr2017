(function () {
  if (typeof window === 'undefined') {
    var $M = require('milsushi2');
  } else {
    var $M = milsushi2;
  }

  console.log('webcl initialization: ' + $M.initcl());

  var tasks = {};
  tasks.task1_pre = function () {
    tasks.a = $M.rand(1000, 1000);
    tasks.b = $M.rand(1000, 1000);
  }

  tasks.task1 = function () {
    var a = $M.gpuArray(tasks.a);
    var b = $M.gpuArray(tasks.b);
    var c = $M.plus(a, b);
    var c_cpu = $M.gather(c);
    a.destruct();
    b.destruct();
    c.destruct();
  };

  tasks.task2_pre = function () {
    tasks.a = $M.rand(1000, 100);
    tasks.b = $M.rand(100, 10);
  };

  tasks.task2 = function () {
    var a = $M.gpuArray(tasks.a);
    var b = $M.gpuArray(tasks.b);
    var c = $M.mtimes(a, b);
    var c_cpu = $M.gather(c);
    a.destruct();
    b.destruct();
    c.destruct();

  };

  tasks.task3_pre = function () {
    tasks.a = $M.rand(1000, 1000);
    tasks.b = $M.rand(1000, 1000);
  };

  tasks.task3 = function () {
    var a = $M.gpuArray(tasks.a);
    var b = $M.gpuArray(tasks.b);
    var c = $M.mtimes(a, b);
    var c_cpu = $M.gather(c);
    a.destruct();
    b.destruct();
    c.destruct();
  };


  tasks.task4_pre = function () {
    tasks.a = $M.rand(200, 500);
    tasks.b = $M.rand(500, 200);
    tasks.c = $M.rand(200, 1);
    tasks.d = $M.rand(200, 500);
    tasks.e = $M.rand(200, 200);
  };

  tasks.task4 = function () {
    $M.autodestruct(function () {
      var a = $M.gpuArray(tasks.a);
      var b = $M.gpuArray(tasks.b);
      var c = $M.gpuArray(tasks.c);
      var d = $M.gpuArray(tasks.d);
      var e = $M.gpuArray(tasks.e);

      var f = $M.mtimes(a, b);
      var clarge = $M.repmat(c, 1, 200);
      var g = $M.plus(f, clarge);
      var g_cpu = $M.gather(g);
      var h = $M.mtimes($M.t(d), e);
      var h_cpu = $M.gather(h);
    });
  };
  tasks.task5_pre = function () {
    tasks.a = $M.rand(1000, 1000);
  }

  tasks.task5 = function () {
    var a = $M.gpuArray(tasks.a);
    var c = $M.log(a);
    var c_cpu = $M.gather(c);
    a.destruct();
    c.destruct();
  };

  if (typeof window === 'undefined') {
    module.exports = { 'sushi_webcl': tasks };
  } else {
    tasks_dict['sushi_webcl'] = tasks;
  }
})();
