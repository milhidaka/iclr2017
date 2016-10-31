#!/bin/bash

#bench_cublas or bench_clblas
CMD=$1

function bench_oneshape() {
  NAME=$1
  M=$2
  N=$3
  K=$4
  #forward
  RESULT=`$CMD $M $N $K N N`
  echo "$NAME,0,$RESULT"
  #backward
  RESULT=`$CMD $M $K $N N T`
  echo "$NAME,1,$RESULT"
  #gradient
  RESULT=`$CMD $K $N $M T N`
  echo "$NAME,2,$RESULT"
}

bench_oneshape conv1_1 802816 64 27
bench_oneshape conv1_2 802816	64	576
bench_oneshape conv2_2 200704	128	576
bench_oneshape conv2_3 200704	128	1152
bench_oneshape conv3_1 50176	256	1152
bench_oneshape conv3_2 50176	256	2304
bench_oneshape conv4_1 12544	512	2304
bench_oneshape conv4_2 12544	512	4608
bench_oneshape conv5_1 3136	512	4608
