This project tries to find out the value of the training data points by comparing it to its distance towards the decision boundary of the classifier trained on the dataset.

Due to the common instinct that the data point which are closer to the boundary has a bigger impact on the classifier, I assume that there should be some sort of correlation between the distance and the value. So I carried out some experiments, which can be found in the following directories.

In the experiments, I used the TMC-Shapley Value as a benchmark valuation method. For linear classifiers, I used the formula of the distance of a point to a hyper-plane. And for non-linear classifiers, I used the idea of finding the 'length' of the minimal perturbation which, when added to the data point, fools the classifier to label the it to a different class.

Experiments on the IRIS dataset and a real world dataset use logistic regression as the linear case, while experiments on the MNIST dataset use LeNet as the non-linear case.