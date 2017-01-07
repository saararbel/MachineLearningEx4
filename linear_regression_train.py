import sys
import numpy as np

def parse_input_file(input_file_path):
    data = np.loadtxt(input_file_path, delimiter= ",")
    Y = data[:, -1]
    X = data[:, :-1]
    X /= np.linalg.norm(X, axis=0)

    X_train = X[:-200]
    X_test = X[-200:]
    Y_train = Y[:-200]
    Y_test = Y[-200:]

    return X_train, Y_train, X_test, Y_test

def cartezian_product(vec1, vec2):
    sum = 0
    for i, e in enumerate(vec1):
        sum += e * vec2[i]
    return sum

def gradient(w, x, y, j):
    num_exmaples = len(x)
    sum = 0.0
    for i in xrange(num_exmaples):
        error = cartezian_product(w,x[i]) - y[i]
        sum = sum + error*x[i][j]

    return (sum / num_exmaples)


def linear_regression(x , y, max_iter, learning_rate):
    w = [0] * x[0].size
    next_w = w
    for i in xrange(max_iter):
        if i % 20 == 0:
            print i
        for j in xrange(len(w)):
            next_w[j] = w[j] - learning_rate * gradient(w,x,y,j)
        w = next_w

    return w


def write_weight_file(w):
    output_file = open("weight.txt", 'w+')
    for n in w:
        output_file.write(str(n) + "\n")


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = parse_input_file(sys.argv[1])
    w = linear_regression(x_train ,y_train ,1000 , 0.1)

    write_weight_file(w)
