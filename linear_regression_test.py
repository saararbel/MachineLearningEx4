import sys
import numpy as np

def parse_input_file(input_file_path):
    data = np.loadtxt(input_file_path, delimiter= ",")
    Y = data[:, -1]
    X = data[:, :-1]
    X /= np.linalg.norm(X, axis=0)

    X_test = X[-200:]
    Y_test = Y[-200:]

    return X_test, Y_test


def parse_weight_file(weight_file_path):
    file = open(weight_file_path, "r")
    w = []
    for line in file:
        w.append(float(line))

    return w

def cartezian_product(vec1, vec2):
    sum = 0
    for i, e in enumerate(vec1):
        sum += e * vec2[i]
    return sum

def test_data(x_test, y_test, w):
    predictions = []
    for idx,xi in enumerate(x_test):
        predictions.append(cartezian_product(xi,w))

    return predictions


def write_predictions_file(predictions):
    file = open("predictions.txt" , 'w+')
    for p in predictions:
        file.write(str(p) + "\n")


if __name__ == '__main__':
    x_test, y_test = parse_input_file(sys.argv[1])
    w = parse_weight_file(sys.argv[2])
    predictions = test_data(x_test,y_test,w)

    write_predictions_file(predictions)