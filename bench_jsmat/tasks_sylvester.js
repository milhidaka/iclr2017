(function () {
  if (typeof window === 'undefined') {
    var $M = require('./sylvester').Matrix;
  } else {
    var $M = Matrix;
  }

  var tasks = {};
  tasks.task1_pre = function () {
    tasks.a = $M.Random(1000, 1000);
    tasks.b = $M.Random(1000, 1000);
  }

  tasks.task1 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = a.add(b);
  };

  tasks.task2_pre = function () {
    tasks.a = $M.Random(1000, 100);
    tasks.b = $M.Random(100, 10);
  };

  tasks.task2 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = a.multiply(b);
  };

  tasks.task3_pre = function () {
    tasks.a = $M.Random(1000, 1000);
    tasks.b = $M.Random(1000, 1000);
  };

  tasks.task3 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = a.multiply(b);
  };


  tasks.task4_pre = function () {
    tasks.a = $M.Random(200, 500);
    tasks.b = $M.Random(500, 200);
    tasks.c = $M.Random(200, 1);
    tasks.d = $M.Random(200, 500);
    tasks.e = $M.Random(200, 200);
  };

  tasks.task4 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = tasks.c;
    var d = tasks.d;
    var e = tasks.e;

    var f = a.multiply(b);
    var g = f.add(c.minor(1, 1, 200, 200));
    var h = d.transpose().multiply(e);
  };

  tasks.task5_pre = function () {
    tasks.a = $M.Random(1000, 1000);
  }

  tasks.task5 = function () {
    var a = tasks.a;
    var c = a.map(function(x){return Math.log(x);});
  };

  if (typeof window === 'undefined') {
    module.exports = {'sylvester': tasks};
  } else {
    tasks_dict['sylvester'] = tasks;
  }
})();
