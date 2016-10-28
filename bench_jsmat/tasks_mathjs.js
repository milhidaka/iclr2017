(function () {
  if (typeof window === 'undefined') {
    var $M = require('./math.min.js');
  } else {
    var $M = math;
  }

  var tasks = {};
  tasks.task1_pre = function () {
    tasks.a = $M.matrix($M.random([1000, 1000]));
    tasks.b = $M.matrix($M.random([1000, 1000]));
  }

  tasks.task1 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.add(a, b);
  };

  tasks.task2_pre = function () {
    tasks.a = $M.matrix($M.random([1000, 100]));
    tasks.b = $M.matrix($M.random([100, 10]));
  };

  tasks.task2 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.multiply(a, b);
  };

  tasks.task3_pre = function () {
    tasks.a = $M.matrix($M.random([1000, 1000]));
    tasks.b = $M.matrix($M.random([1000, 1000]));
  };

  tasks.task3 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = $M.multiply(a, b);
  };


  tasks.task4_pre = function () {
    tasks.a = $M.matrix($M.random([200, 500]));
    tasks.b = $M.matrix($M.random([500, 200]));
    tasks.c = $M.matrix($M.random([200, 1]));
    tasks.d = $M.matrix($M.random([200, 500]));
    tasks.e = $M.matrix($M.random([200, 200]));
  };

  tasks.task4 = function () {
    var a = tasks.a;
    var b = tasks.b;
    var c = tasks.c;
    var d = tasks.d;
    var e = tasks.e;

    var f = $M.multiply(a, b);
    //repeat c horizontally
    var clist = [];
    for (var i = 0; i < 200; i++) {
      clist.push(c);
    }
    var clarge = $M.concat.apply($M, clist);
    var g = $M.add(f, clarge);
    var h = $M.multiply($M.transpose(d), e);
  };

  tasks.task5_pre = function () {
    tasks.a = $M.matrix($M.random([1000, 1000]));
  }

  tasks.task5 = function () {
    var a = tasks.a;
    var c = $M.log(a);
  };

  if (typeof window === 'undefined') {
    module.exports = { 'mathjs': tasks };
  } else {
    tasks_dict['mathjs'] = tasks;
  }
})();
