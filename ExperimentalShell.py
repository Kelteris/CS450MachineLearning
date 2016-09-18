###########################################
# Robert Dickerson
# Brother Burton
# CS 450 Machine Learning
###################################################
#Klenton Stone gave me some pointers on how to properly split the data
import random
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
iris = datasets.load_iris()

class DefaultHardCode:


    def train(self, data_set, target_set):
        print("\nThe system has been trained on the new set of data.\n")


    def predict(self, data_set):
        x = []
        for i in data_set:
            x.append(0)
        return x


def get_accuracy(results_of_predict, test_targets):
    value_correct = 0
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]

    print("The system correctly predicted ", value_correct, " of ", test_targets.size,
          ". \nThe system was able to correctly predict ",
          "{0:.2f}% of the time!".format(100 * (value_correct / test_targets.size)), sep="")

def train_system():
    random.shuffle(iris.data)
    testAmount = float(0.3)
    timesShuffled = 15
    train_data, test_data, train_target, test_target = tsp(iris.data, iris.target, test_size = testAmount,
                                                           random_state = timesShuffled)
    classifier = DefaultHardCode()
    classifier.train(train_data, train_target)
    get_accuracy(classifier.predict(test_data), test_target)


train_system()