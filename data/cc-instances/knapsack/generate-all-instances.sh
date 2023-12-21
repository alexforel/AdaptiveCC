#!/bin/bash

inputPath="../../orlib/"
files=("instance1.txt" "instance2.txt" "instance3.txt" "instance4.txt" "instance5.txt" "instance6.txt" "instance7.txt" "instance8.txt")
scenarios=(500 1000 3000 5000)
instanceQty=5

for instanceFile in "${files[@]}"
do
    for scenarioQty in "${scenarios[@]}"
    do
	python instance-generator.py $"${inputPath}${instanceFile}" $scenarioQty $instanceQty
    done
done
