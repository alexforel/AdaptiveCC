files=("ccmknap-10-10" "ccmknap-20-10" "ccmknap-40-30")
scenarios=(500 1000 3000 5000)
indexes=5
continuous_vars=(0 1)
epsilons=(0.1 0.2)
methods=(1 2 3 4)

main_path="./main.py"
input_path="./data/cc-instances/knapsack/"
output_path="./results/tables/"

for file in "${files[@]}"
	do
	for scenario in "${scenarios[@]}"
		do
		for index in $(seq 1 $indexes)
			do
			for continuous_var in "${continuous_vars[@]}"
				do
				for epsilon in  "${epsilons[@]}"
					do
					for method in "${methods[@]}"
						do
						file_name="${input_path}${file}-$scenario-${index}.csv"
						python $main_path $file_name $continuous_var $epsilon $method $output_path
						wait
						sleep 2
					done
				done
			done
		done
	done
done