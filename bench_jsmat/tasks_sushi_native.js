(function () {
  if (typeof window === 'undefined') {
    var $M = require('milsushi2');
  } else {
    var $M = milsushi2;
  }

  var tasks = {};
  tasks.task1_pre = function () {
    tasks.a = $M.rand(1000, 1000);
    tasks.b = $M.rand(1000, 1000);
  }

  tasks.task1 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.plus(a, b);
  };

  tasks.task2_pre = function () {
    tasks.a = $M.rand(1000, 100);
    tasks.b = $M.rand(100, 10);
  };

  tasks.task2 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.mtimes(a, b);
  };

  tasks.task3_pre = function () {
    tasks.a = $M.rand(1000, 1000);
    tasks.b = $M.rand(1000, 1000);
  };

  tasks.task3 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.mtimes(a, b);
  };


  tasks.task4_pre = function () {
    tasks.a = $M.rand(200, 500);
    tasks.b = $M.rand(500, 200);
    tasks.c = $M.rand(200, 1);
    tasks.d = $M.rand(200, 500);
    tasks.e = $M.rand(200, 200);
  };

  tasks.task4 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = tasks.c;
    var d = tasks.d;
    var e = tasks.e;

    var f = $M.mtimes(a, b);
    //repeat c horizontally
    var clarge = $M.repmat(c, 1, 200);
    var g = $M.plus(f, clarge);
    var h = $M.mtimes($M.t(d), e);
  };

  tasks.task5_pre = function () {
    tasks.a = $M.rand(1000, 1000);
  }

  tasks.task5 = function () {
    var a = tasks.a;
    var c = $M.log(a);
  };

  if (typeof window === 'undefined') {
    module.exports = { 'sushi_native': tasks };
  } else {
    tasks_dict['sushi_native'] = tasks;
  }
})();
