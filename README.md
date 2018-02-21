# Dog Breed Classifier
---

In this project I used Deep Learning and Computer Vision techniques for identifying the breed of a dog presented on image. 

The goal was to build a pipeline which gives high accuracy predicting the canine’s breed. 
Then wrap the code into web application and finally deploy it to the public web server.

---

At the moment, the application is available [here](http://alobov.pythonanywhere.com/). 
And the output looks something like this:

![Gray image example](/images/example.jpg)

_It can also classify human images, give it a try!_

---

The first goal was achieved by using transfer learning. I took the model with superhuman performance aka ResNet50, 
cut off the top layer, froze the weights and added a layer with only 133 outputs which are supposed to represent canine's breeds. 
Then I trained this new network on a few dataset to get new weights for the top layer. 
It gave me 82% of accuracy in very few epochs what was quite enough.

For the web framework i have chosen Flask which is really easy to understand yet a very powerful tool. 

Then I tried to deploy it on github pages, but it didn't want to work for some reason. So I deployed it on pythonanywhere.com

--- 

## List of possible improvements

Although all of the project goals are achieved, there is alot of stuff to do here. But since this app was made within a Deep Learning Nanodegree the main work is in improving the accuracy of predictions . 
* Prediction improvements:
  - Data Augmentation
  - Functionality for dog mutts
  - Finding a better face detector
  - Classify several types of dogs and humans
* Another functional improvements:
  - Overlay dog ears and nose on human face
  - Improve interface robustness
* Visual improvements:
  - Design background, inputs and all stuff
  - More accommodative 'processing' label
  - and so on and so forth
