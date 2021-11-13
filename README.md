In this homework, you implement parametric classification where class densities are assumed to be
Gaussian.
You are provided with two dataset files. As their names imply, the training.csv file will be used for
training and the testing.csv file will be used for testing. Each row of the files corresponds to one
instance. You have 150 instances per file. The first column of the row contains the integer value
representing the input x and the second column contains class C label (1, 2 or 3) of that instance.
You should use the training instances to estimate the parameters, which are the class priors and the
parameters of the Gaussian densities, namely means and variances. The test instances, unused
during training, is used to estimate the generalization performance of your model.
<b> 1) </b> Plot the likelihood and posterior distributions, together with the training and test data instances
on the same plot. Use different colors/symbols for different distributions and classes.
2) Calculate the performance of the model:
a) Assume 0/1 loss and calculate the 3x3 confusion matrices of your model on both on the
training and the test instances.
b) Assuming that the cost (loss) of assigning an input to a wrong class is 4 and the cost of
rejecting an instance is 1, calculate the threshold of decision for minimum expected risk and
calculate the 4x3 confusion matrices on training and test sets.
