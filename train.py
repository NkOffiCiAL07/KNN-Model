import numpy as np
import csv
import sys
import math
from validate import validate


#importing data from environment
def import_data(test_X_file_path, test_Y_file_path = "train_Y_knn.csv"):
    X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(test_Y_file_path, delimiter=',', dtype=int)
    return X, Y

"""
Returns:
ln norm distance
"""
def compute_ln_norm_distance(vector1, vector2, n):
    sum = 0
    for i in range(len(vector2)):
        sum = sum + abs((vector2[i]-vector1[i])**(n))
    return (sum)**(1/n)

"""
Returns:
Indices of the 1st k- nearest neighbors in train_X, in the order with nearest first.
"""
def find_k_nearest_neighbors(train_X, test_example, k, n):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
        distance = compute_ln_norm_distance(train_elem_x, test_example, n)
        indices_dist_pairs.append([index, distance])
        index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


"""
Returns:
Classified points using knn method
"""
def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    classified_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n)
        top_knn_labels = []

        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        Y_values = list(set(top_knn_labels))

        max_count = 0
        most_frequent_label = -1
        for y in Y_values:
            count = top_knn_labels.count(y)
            if(count > max_count):
                max_count = count
                most_frequent_label = y

        classified_Y.append(most_frequent_label)
    return np.array(classified_Y)

"""
Returns:
Calculates accuracy of the model.
"""
def calculate_accuracy(predicted_Y, actual_Y):
    count = 0
    for i in range(len(predicted_Y)):
        if (predicted_Y[i] == actual_Y[i]):
            count+=1
    return (count/len(actual_Y))


"""
Returns K value based on validation data.
"""
def k_value_using_validation_set(train_X, train_Y, validation_split_percent, n):
    total_num_of_observations = len(train_X)
    train_length = math.floor((100 - validation_split_percent)/100 * total_num_of_observations )
    validation_X = train_X[train_length :]
    validation_Y = train_Y[train_length :]
    train_X = train_X[0 : train_length]
    train_Y = train_Y[0 : train_length]
 
    best_k = -1
    best_accuracy = 0
    for k in range(1, train_length+1):
        predicted_Y = classify_points_using_knn(train_X,train_Y, validation_X, n, k)
        accuracy = calculate_accuracy(predicted_Y,validation_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k


"""
Driver function.
"""
if __name__ == "__main__":
    X, Y = import_data(test_X_file_path = "train_X_knn.csv", test_Y_file_path = "train_Y_knn.csv")
    best_K_value = k_value_using_validation_set(X, Y, validation_split_percent = 70, n = 2)
    print("Best value of K using cross validation : ", best_K_value)