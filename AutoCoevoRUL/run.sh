#!/bin/bash
for (( c=0; c<10; c++ ))
do
  cp conf/experiments/experiments.cnf.template conf/experiments/experiments.cnf
  sed -i -e "s/<SEED>/$c/g" conf/experiments/experiments.cnf
  ./run_autocoevorul.sh
done