import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

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
        prev_theta = theta
        grd = gradient(x, y, theta) * alpha
        theta = theta - grd
        if (i % int(max_iter / 25)) == 0:
            print(theta)
        if i > max_iter / 4 and theta[0] == prev_theta[0] and theta[1] == prev_theta[1]:
            break
    return theta

def normalize(array):
    ret = np.empty([])
    minElem = min(array)
    maxElem = max(array)
    for elem in array:
        ret = np.append(ret, (elem - minElem) / (maxElem - minElem))
    return (ret[1:])

def denormalize(array, elem):
    return ((elem * (max(array) - min(array))) + min(array))

def disp_graph(x, y, theta):
    #Matplotlib

    fx = [min(x), max(x)]
    fy = []
    for elem in fx:
        elem = theta[1] * ((elem - fx[0]) / (fx[1] - fx[0])) + theta[0]
        fy.append(denormalize(y, elem))
     np.set_printoptions(suppress=True)
    print(np.polyfit(fx, fy,1))
    plt.plot(x, y, "+r")
    plt.plot(fx, fy, "-b")
    plt.xlabel("Miles")
    plt.ylabel("Price")
    plt.show()

def save_theta(theta):
    try:
        np.savetxt("theta.csv", theta.reshape(1, -1), delimiter=',', header="theta0,theta1", fmt="%1.8f")
    except:
        sys.exit("ft_linear_regression: Error saving theta.csv")

def get_dataset():
    if len(sys.argv) != 2:
        sys.exit("ft_linear_regression: Wrong argument number")
    try:
        data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    except:
        sys.exit("ft_linear_regression: Unable to open file")

    return (data[:,0], data[:,1], normalize(data[:,0]), normalize(data[:,1]))

if __name__ == "__main__":
    miles, price, x, y = get_dataset()
    #Run linear_regression
    theta = fit_(x, y, np.array([1, 1]), 1, 2000)
    save_theta(theta)
    disp_graph(miles, price, theta)
