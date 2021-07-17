import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
plt.style.use("seaborn-whitegrid")
warnings.filterwarnings("error")

def gradient(x, y, theta):
    m = x.shape[0]
    x = x.reshape(-1, 1)
    y = y.reshape(-1,)
    ones = np.ones(m).reshape(-1, 1)
    xp = np.hstack((ones, x))
    prediction = (np.dot(xp, theta) - y)
    nabla = np.dot(xp.transpose(), prediction)
    return (nabla / m)

def fit_(x, y, theta, alpha, max_iter):
    for i in range(max_iter):
        grd = gradient(x, y, theta) * alpha
        print(grd)
        try:
            theta = theta - grd
        except RuntimeWarning:
            print("Warning at {}".format(i))
            print(theta)
            print(grd)
            sys.exit("qqq")
    return theta

if __name__ == "__main__":
    #Check arg
    if len(sys.argv) != 2:
        sys.exit("Wrong argument number")

    #CSV to np.array
    data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    x = data[:,0]
    y = data[:,1]
    print(x)
    print(y)

    #Run linear_regression
    theta = fit_(x, y, np.array([1, 1]), 5e-8, 15)

    #save
    try:
        np.savetxt("theta.csv", theta.reshape(1, -1), delimiter=',', header="theta0,theta1", fmt="%1.8f")
    except:
        sys.exit("Error saving theta.csv")

    #Matplotlib
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(x, y, ".r")

    f = lambda var: theta[1] * var + theta[0]
    xx = np.linspace(x.min(), x.max(), num=100)
    plt.plot(xx, f(xx))

    plt.show()
