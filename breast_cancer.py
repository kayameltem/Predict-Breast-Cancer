#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np # 1.16.5
import pandas as pd # 0.24.2
import matplotlib.pyplot as plt# 2.2.3
#Python 2.7 kullandım

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )

def read_and_divide_into_train_and_test(csv_file):
    file = pd.read_csv(csv_file, na_values = [ "?"])
    median = file['Bare_Nuclei'].median()
    file['Bare_Nuclei'].fillna(median, inplace = True)
    training_inputsx = file.iloc[:560, 1:-1]
    corrmatrix = training_inputsx.corr()
    pos = np.arange(len(corrmatrix.columns))
    plt.xticks(pos,training_inputsx.columns, rotation= 90)
    plt.yticks(pos,training_inputsx.columns)
    plt.imshow(corrmatrix, cmap='hot' ,interpolation='nearest')
    plt.ylabel("Correlation")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    ra = np.genfromtxt(csv_file, dtype = int ,delimiter = ",")
    training_inputs = ra[1:560, 1:-1]
    training_labels = ra[1:560,-1]
    training_labels = training_labels.reshape(training_labels.shape[0],-1)
    test_inputs = ra[560:,1:-1]
    test_labels =ra[ 560:,-1]
    test_labels = test_labels.reshape(test_labels.shape[0],-1)

    return training_inputs, training_labels, test_inputs, test_labels

def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    test_prediction = sigmoid(np.dot(test_inputs, weights))
    test_predictions = np.round(test_prediction)
    test_outputs = test_labels 

    for predicted_val, label in zip(test_predictions, test_outputs):
        if predicted_val == label:
            tp += 1
    accuracy = tp /float(test_inputs.shape[0])
    return accuracy

def plot_loss_accuracy(accuracy_array, loss_array):
    plt.plot(accuracy_array)
    plt.xlabel("Accuracy Change")
    plt.show()
    plt.plot(loss_array)
    plt.xlabel("Loss Change")
    plt.show()

def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        input = training_inputs
        training_outputs = np.dot(input, weights)
        training_outputs = sigmoid(training_outputs) #calculate outputs
        loss = training_labels - training_outputs#calculate loss
        tuning = loss * sigmoid_derivative(training_outputs) #calculate tuning
        weights = weights + np.dot(np.transpose(input),tuning)#update weights
        loss_array.append(np.mean(loss))
        accuracy_array.append(run_on_test_set(test_inputs,test_labels,weights))

    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()
