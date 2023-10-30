#!/bin/sh

pyDir="${PWD}"
pyPath="$pyDir/1b_generate_mave_global_inter.py"

for i in {0..50};
do
	foo=$(printf "%02d" $i)
	echo "${foo}"
	python $pyPath "${foo}"
done
