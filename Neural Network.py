###########################################
# Robert Dickerson
# Brother Burton
# CS 450 Machine Learning
###################################################
import random
import sys
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp

class NeuralNode:
    # A bias node will always be included in the NeuralNode
    bias = -1
    def __init__(self, numberOfInputs):
        self.weights = []
        # add a one extra iteration for the bias node
        for i in range(numberOfInputs + 1):
            self.weights.append(random.random())

    def neuronScore(self, inputs):
        # append the bias node to the list of inputs
        # inputs.append(self.bias)
        # inputs.append(-1)
        score = 0.0
        for i in range(len(inputs)):
            score += inputs[i] * self.weights[i]
        score += self.bias * self.weights[len(inputs) + 1]
        if score >= 0:
            return True
        else:
            return False


class NeuralNetwork:
    # creating a network that can hold the layers and each node inside them
    def __init__(self):
        self.layers = []

    def createNetwork(self, numberOfNeurons, numberOfInputs):
        neurons = []
        for _ in range(numberOfNeurons):
            neurons.append(NeuralNode(numberOfInputs))
        self.layers.append(neurons)

    #Kory gave me some pointers on calculating the final result
    def calcTargets(self, data):

        finalResults = []
        for iCurrentLayer in range(len(self.layers)):
            for numInputs in range(len(data)):
                targets = []
                for node in range(len(self.layers[iCurrentLayer])):
                    # used a multi dimensional array to hold the nodes with their respective layers
                    neuron = self.layers[iCurrentLayer][node]
                    targets.append(neuron.neuronScore(data[numInputs]))
                finalResults.append(targets)
        return finalResults



def get_accuracy(results_of_predict, test_targets):
    value_correct = 0
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]

    print("The system correctly predicted ", value_correct, " of ", test_targets.size,
          ". \nThe system was able to correctly predict ",
          "{0:.2f}% of the time!".format(100 * (value_correct / test_targets.size)), sep="")

def train_system(data, target, classifier):

    testAmount = float(0.3)
    timesShuffled = 15


    train_data, test_data, train_target, test_target = tsp(data, target, test_size = testAmount,
                                                           random_state = timesShuffled)

    #classifier.train(train_data, train_target)

    get_accuracy(classifier.predict(train_data, train_target, test_data), test_target)

def main(argv):


    #inputs = []
    number = 0

    nn = NeuralNetwork()
    while number != 1 or number != 2:
        print ("\nChoose the Data you would like to use\n"
               "To view iris Prediction,       enter 1\n"
               "To view diabetes Prediction    enter 2")

        number = int(input("Choice: "))

        if (number == 1):
            irisData = datasets.load_iris()
            #random.shuffle(irisData)
            trainData = irisData.data
            targetData = irisData.target
            #train_system(trainData, targetData, nn)
            nn.createNetwork(3, len(trainData))

        if (number == 2):
            diabetesData = datasets.load_diabetes()
            #random.shuffle(diabetesData)
            trainData = diabetesData.data
            targetData = diabetesData.target
            #train_system(trainData, targetData, nn)
            nn.createNetwork(2, len(trainData))

        testResults = nn.calcTargets(trainData)
        print(testResults)



if __name__ == "__main__":
    main(sys.argv)