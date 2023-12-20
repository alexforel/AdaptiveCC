import csv
import sys
import numpy as np

args = sys.argv[1:]

fileLocation = args[0]
qtyScenarios = int(args[1])
qtyInstances = int(args[2])

# PARAMETERS FOR INSTANCES
removeNegatives = True
withFailures = True

# READ & EXTRACT

# Read instance data
with open(fileLocation, newline='\n') as csvfile:
    spamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    completeFile = [row for row in spamReader]

# Extract data and save in relevant files
instanceInfo = np.array(completeFile[0][0:2]).astype(int)
qtyItems = instanceInfo[0]
qtyConstraints = instanceInfo[1]
objParam = np.array(completeFile[1]).astype(float)
matrixA = np.array(completeFile[2:2+qtyConstraints]).astype(float)
vectorB = np.array(completeFile[2+qtyConstraints]).astype(float)

# GENERATE DATA
for instance in range(1, qtyInstances+1):

    # Generate failure probability with exp distribution
    probaFail = np.random.default_rng().exponential(scale=0.1, size=qtyItems)
    probaFail[probaFail < 0] = 0
    probaFail[probaFail > 1] = 1

    # Generate failures in scenarios
    failuresScenario = np.full((qtyItems, qtyScenarios), 1)
    for i in range(qtyItems):
        failuresScenario[i, :] = failuresScenario[i, :] - \
            np.random.binomial(1, probaFail[i], qtyScenarios)
    failuresScenario = failuresScenario.T

    # Generate matrix A for each scenario
    randomMatrixA = np.zeros((qtyScenarios, qtyConstraints, qtyItems))
    for i in range(qtyConstraints):
        for j in range(qtyItems):
            if withFailures:
                randomSamples = np.random.normal(
                    matrixA[i, j], abs(matrixA[i, j]*0.1),
                    qtyScenarios)*failuresScenario[:, j]
            else:
                randomSamples = np.random.normal(
                    matrixA[i, j], abs(matrixA[i, j]*0.1), qtyScenarios)
            if removeNegatives:
                randomSamples[randomSamples < 0] = 0
            randomMatrixA[:, i, j] = randomSamples
    randomMatrixA = np.round(randomMatrixA, 4)

    # PRINT TO FILE
    instanceFile = "ccmknap-" + str(qtyItems) + "-" + \
        str(qtyConstraints) + "-" + str(qtyScenarios) + "-" + \
        str(instance)+".csv"
    print(instanceFile)
    with open(instanceFile, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(np.insert(instanceInfo, 2, qtyScenarios))
        writer.writerow(objParam)
        for s in range(qtyScenarios):
            writer.writerows(randomMatrixA[s, :, :].tolist())
        writer.writerow(vectorB)
