README
====================
NeuralNet.py

Code should be fairly self-explanatory. Use the train or KFold methods to test the code. Their inputs should be self explanatory. Parameters are global values and can be found on line 108.

====================
NearestNeighbor_Match.py and NearestNeighbor_Output.py

NearestNeighbor_Match computes the closest mnist match to each image in the dataset. 
NearestNeighbor_Output takes these matches to output useable boundary boxes or their simply their sums. 

You can adjust which images they are working on in the main method. Simply run the code and they will produce csv files with their outputs. This code is parallelized but still quite slow. Both these methods use MNIST in csv form, as Yann LeCun's website was down at the time we wanted those images. The MNIST CSV files are too large to submit but are available here: http://pjreddie.com/projects/mnist-in-csv/

====================
sift_logsitic.py

Code should be self explanatory. Just run python script to see results.

====================
cnn.zip

Unzip cnn.zip to see all the files for the cnn. Includes a second readme.