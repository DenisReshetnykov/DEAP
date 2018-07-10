from sklearn.metrics import confusion_matrix



def calculateAccuracy(predicted_data, test_data):
    accuracy = 0.0
    if len(predicted_data) == len(test_data):
        for index in range(len(predicted_data)):
            if predicted_data[index] == test_data[index]:
                accuracy += 1
        accuracy /= len(predicted_data)
    return accuracy