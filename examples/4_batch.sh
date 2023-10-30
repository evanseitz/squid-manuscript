#!/bin/sh

pyDir="${PWD}"
pyPath="$pyDir/4_analyze_outputs.py"

# the below list {,,,} must start with '0' to initiate recording all mean-error values to file; 
# also, 'max_flank' must match the last entry in the list

declare -i max_flank=100

declare -a GaugeArray=("hierarchical" "empirical" "wildtype" "default") #gauges to loop over
#declare -a GaugeArray=("hierarchical") #gauges to loop over

for gauge in ${GaugeArray[@]};
do
	for i in {0,5,10,15,20,30,40,50,60,70,80,90,100}; #number of flanking nt (per side);
	#for i in {0,50}; #required in code to always keep zero in list
	do
		foo=$(printf "%02d" $i)
		echo "${foo}"
		python $pyPath "${foo}" "${max_flank}" "${gauge}"
	done
done
