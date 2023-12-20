# Adaptive Partitioning for Chance-Constrained Problems with Finite Support
This code can be used to reproduce all results in the paper titled "Adaptive Partitioning for Chance-Constrained Problems with Finite Support by Marius Roland, Alexandre Forel, and Thibaut Vidal.

[Paper available here](https://optimization-online.org/2023/12/adaptive-partitioning-for-chance-constrained-problems-with-finite-support/)   |   [Short video presentation](https://youtu.be/KMPqVof2k2U?feature=shared)


## Installation
The project requires the Gurobi solver to be installed with an authorized license. Free academic licenses can be obtained at: https://www.gurobi.com/academia/academic-program-and-licenses/ .

The packages required are listed in the file `requirements.txt`. Run the following commands from the project root to install the requirements. You may have to install python and venv before.

```shell
    virtualenv -p python3.10 env
    source env/bin/activate
    pip install -r requirements.txt
    python -m pip install -i https://pypi.gurobi.com gurobipy
    python setup_violations.py build_ext --inplace
```

The installation can be checked by running the test suite:
```shell
   python -m pytest
```

## Content
The main scripts that run experiments and process the results are in the root folder. The folders have the following contents:
* `data`: deterministic instances of multi-dimensional knapsack problems, run the bash script `generate-all-instances.sh` to generate chance-constrained instances,
* `src`: all methods and classes needed to generate and analyze the experimental results,
* `tests`: a set of unit and integration tests.

## How to reproduce the paper results
All experiments are run using the 'main.py' script and providing the relevant arguments. This can be done using a bash script. For instance, the following runs all experiments used in the paper:
```shell
files=("ccmknap-10-10" "ccmknap-20-10" "ccmknap-40-30")
scenarios=(500 1000 3000 5000)
indexes=5
continuous_vars=(0 1)
epsilons=(0.1 0.2)
methods=(1 2 3 4)

main_path="./cclp-adaptive/main.py"
input_path="./cclp-adaptive/data/cc-instances/knapsack/"
output_path="./cclp-adaptive/results/tables/"

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
						python main.py $main_path $file_name $continuous_var $epsilon $method $output_path
						wait
						sleep 2
					done
				done
			done
		done
	done
done

```

Run the script `export_results` to generate the csv files used to create the tables and figures presented in the paper.