import sys
import numpy as np

def cost_prediction(x, theta):
    return (theta[0] + theta[1] * x)


if __name__ == "__main__":
    #args checking
    if len(sys.argv) != 2:
        sys.exit("usage: python prediction.py kilometers")
    try:
        kilometers = float(sys.argv[1])
    except:
        sys.exit("Argument must be numeric")

    #Reading data file
    try:
        theta = np.genfromtxt("theta.csv", delimiter=',', skip_header=1)
    except:
        sys.exit("theta.csv error")

    #Executing function and returning predicted value
    print(cost_prediction(kilometers, theta))

