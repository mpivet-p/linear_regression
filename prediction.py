import sys
import numpy as np

def cost_prediction(x, theta):
    return (theta[0] + theta[1] * x)


if __name__ == "__main__":
    #Reading data file
    try:
        theta = np.genfromtxt("theta.csv", delimiter=',', skip_header=1)
    except:
        sys.exit("theta.csv error")

    try:
        kilometers = float(input("please enter a distance to get a price prediction: "))
    except:
        sys.exit("Argument must be numeric")

    #Executing function and returning predicted value
    print(cost_prediction(kilometers, theta))

