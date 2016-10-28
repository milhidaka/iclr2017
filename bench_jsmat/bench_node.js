'use strict';

function load_bench_objs() {
  var tasks_dict = {};
  Object.assign(tasks_dict, require('./tasks_sylvester'));
  Object.assign(tasks_dict, require('./tasks_mathjs'));
  Object.assign(tasks_dict, require('./tasks_sushi_native'));
  Object.assign(tasks_dict, require('./tasks_sushi_webcl'));
  return tasks_dict;
}

function calc_task(tasks, name) {
  var task_pre = tasks[name + '_pre'];
  var task = tasks[name];
  task_pre();
  var begin = new Date();
  task();
  var end = new Date();

  return (end-begin);
}

function calc_stat(tasks, name, times, lib_name) {
  calc_task(tasks, name);//first execution to load code
  var time_sum = 0;
  for (var i = 0; i < times; i++) {
    time_sum += calc_task(tasks, name);
  }
  var time_avg = time_sum / times;
  console.log(lib_name + ',' + name + ',' + time_avg);
}

function main() {
  var tasks_dict = load_bench_objs();
  var task_names = ['task1', 'task2', 'task3', 'task4', 'task5'];
  for (var lib_name in tasks_dict) {
    var tasks = tasks_dict[lib_name];
    for (var i = 0; i < task_names.length; i++) {
      var task_name = task_names[i];
      calc_stat(tasks, task_name, 5, lib_name);
    }
  }
}

main();
