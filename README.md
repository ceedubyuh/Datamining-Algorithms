# Datamining-Algorithms
## Collection of programs I wrote for a datamining course
Unfortunately I no longer have access to the original data files associated with each program, but the code itself is too good to not share. 

### Food_Pantry_Data.py:
This project was a group final project built to help visualize the usage of our campuses food bank aka Food Pantry. We took a picture of the banks database format and created a mock data file .CSV in order
to test our program with. We split the data into Before, During and After Covid-19 lockdown data and with that data, we tried to predict how the numbers would equalize after the lockdown had fully subsided.

### Handwriting_Recognition_by_KNN.py:
This project I wrote alone using a K-nearest-neighbor algorithm to scan pictures of handwritten numbers with a set of computer generated numbers and use inferences to determine what number the program saw
with the handwritten set. I was able to achieve above a 60% accuracy rating from the handwritten set, but I am sure the data wasn't as accurate because I wrote my numbers on paper much smaller and thinner 
than I had liked.

### Datamining_Cluster.py:
Using Hierarchical clustering, this program was written to take in a large version of the Iris dataset, specifcally the sepal lengths, and cluster the min and max distances of them all to predict the unknown
lengths.

### irisentropy.py:
Using Nodes and a Decision Tree, this program also utilized the iris dataset to predict if the unknown flowers were Iris' or not. This version also employed entropy for the first time, meaning the Decission Tree
was able to prune itself for more desirable results with up to a 96% accuracy rating.
