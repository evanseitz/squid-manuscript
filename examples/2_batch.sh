#!/bin/sh

pyDir="${PWD}"
pyPath="$pyDir/2_generate_mave.py"

#for i in 5; #model a single sequence; positive integer >= 0
#for i in {0,1}; #model a specific list of sequences; remove spaces between commas as shown
for i in {0..50}; #model a consecutive series of sequences
do
	foo=$(printf "%02d" $i)
	echo "${foo}"
	python $pyPath "${foo}"
done
