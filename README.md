# Adaptive Partitioning for Chance-Constrained Problems with Finite Support
![Tests badge](https://github.com/alexforel/AdaptiveCC/actions/workflows/main.yml/badge.svg?branch=main)

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
All experiments are run using the `main.py` script and providing the relevant arguments. This can be done using a bash script. For instance, the script `run_all_experiments.sh` can be used to run all experiments presented in the paper. Run the script `export_results` to read the experiment results and generate the csv files used to create the tables and figures.
