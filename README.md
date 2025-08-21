# Attack and Defense against traffic light classification

This project demonstrates the vulnerabilities in a self-driving system, focusing on traffic light detection and classification. 
This project will take you through regular classification of traffic lights that is done with a convoluted neural network with 6 trainable layers and 6 non-trainable layers, such as normalisation and maxpooling. 
Following the training and explanation of the clean model. It will be attacked with a data poisoning backdoor attack consisting of triggers implemented with pink squares representing real-world Post-It notes.


Attack demonstation 
------
Here we can see a trojan horse attack where when a trigger is present, in this case a pink postit sticker, the model incorrectly classifies the traffic lights.
As can be seen in the top left corner, the model classifies red lights as green ones with high confidence.

![Attack Demo](./Attack.gif)




Defense demonstration
------
In this example, one of the developed defense methods were applied.
As can be seen in the top left corner, the model now correctly classifies red lights as red.

![Defense Demo](./Defense.gif)


### The project code is located in this [Jupyter Notebook](https://github.com/Klattraren/Vulnerabilities-in-Autonomous-Drive-Systems-Focusing-on-Traffic-Light-Classification/blob/main/Vulnerabilities-in-AD_systems_project/Traffic_Light_Detection_Project.ipynb)
