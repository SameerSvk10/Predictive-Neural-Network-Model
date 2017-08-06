# Predictive-Neural-Network-Model
Model to predict whether a person X will buy a product Y or not.

Problem Statement:
The task is to predict the probability that a person X will buy a product Y. There are a bunch of largely demographic related features available about person X. There are also features available around X’s past activities. We also have aggregated data about people who typically buy product Y available as features (their demographics and past activities). We however do not know which features are which with certainty. The variable C tells us if the person X actually bought the product Y. Please provide a program/script that can take a tab delimited file with these 22 features and output a file with the index and the predicted output (1 or 0).

ClassificationProblem1.txt is the tab-delimated file containing 1, 01,180 samples where each sample has 22 features and 'C' variable(ttarget variable) which tells if the person X actually bought the product Y(1) or not(0). train_model.py is a script used to train the neural network model with training data. 'testing.py' is a script which takes 'Classification1Test.txt' as input data output a file with predicted output(1 or 0) for each sample. 'my_model1.h5' is a trained model with training data. You can directly load the model in 'testing.py' script. 'test_target_pred.txt' is the output file for the given test dataset.
