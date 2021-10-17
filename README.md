# Phase_5 
# Recognizing Human Facial Expressions 

## Description :

The goal of the project was to build a Convolutional Neural Network (CNN) model
to classify human facial expressions, and use this model in special glasses for 
children with autism to help improve their ability to recognize human emotions 
through people's facial expressions. 
Using CNN for image recognition can be applied as part of the diagnosis procedures
to assist medical professionals in diagnosing and detecting illnesses or abnormalities
in X-rays and scans of all kinds.  

## EDA :

The data was obtained from Kaggle’s website, which was divided into training (80%),
testing (10%), and validation (10%). The images came in ImageFolder format. 
The data is labeled and has 7 classes (0- Angry,1- Disgust, 2 - Fear, 3 - Happy,
4 - Sad, 5 - Surprise, 6 - Neutral). The images in the data came in different
pixel sizes so I resized them all to one size, 48x48 pixels. Rescale the images 
using Keras using ImageDataGenerator.Split the data to X, and target Y , using (.next) with ImageDataGenerator.

## MODELING :

### BaseModel:


To establish a baseline model I followed architectural principles of the VGG models.
And I did that because the architecture is easy to understand and it won best performance
in the ILSVRC 2014 competition. The architecture is made out of blocks, and each block 
has two CNN layers stacked with 3x3 filters and a max pooling layer. Each time we add a new block we increase the number of nodes in the layer like (32, 64, 128, 256, 512, 1024). 

Then we ran our data through the model starting with one block, then two blocks. The model with one block had a validation accuracy of 50%, the accuracy increased by 3% when we added the second block, which showed us that the deeper the structure of the model the better the model performance. Then I decided to go up to five blocks and as expected the model performed even better and the accuracy increased by another 5%. There was a huge gap between our training accuracy and our validations accuracy though, which was an indication of model overfitting that needed to be addressed. To address the overfitting I decided to try out four different Regularization techniques and compare the result to see which one would work best. 


### Regularization : 

#### Weight Regularization :

The first regularization technique I tried was weight regularization. It works by updating the loss function to penalize the model in proportion to the size of the model weights. By making the model weights smaller we made the model more stable and general. The result did not look good, and accuracy went down 1% and the model still had an overfitting issue. 

#### Data Augmentation :

The second regulation technique I tried was data augmentation. This technique modifies the training data we have by adding variety to our data without having to add new data. I augmented the data with small shifts in the height and width of the images, and included a horizontal flip. The reason for the small amount changes is because the images are small to start with so I did not want to lose useful features in the images. The accuracy results dropped down to 48% which was expected because the images were very small to start with, and losing out on any feature did not help the model.

#### Droput :

The third regulation technique I tried was drop out. A drop out technique randomly picks neurons to ignore throughout the training data in each hidden layer, which helps slow down the learning rate while increasing the accuracy and decreasing the loss. I tried a 20% drop out rate for all the blocks and the Dense layer, which did help the accuracy, but we still had an issue with over fitting. So I tried a drop out rate of 30% which showed no sign of overfitting and got an accuracy rate of 62%. 

#### Drop out 30% + Additional Layer/ 6 Blocks :

I thought of trying to use the regulation technique that worked best, which is drop out with 30% dropout rate, combined with a 6th block added to the base model since it showed that adding additional layers for the model helps the accuracy, while giving the model more time to train by increasing the Epochs number from 100 to 400. This resulted in an accuracy of 66%, which was the highest accuracy I got, but the model ended up over-fitting again.

#### Drop out 30% + Additional Layer/ 6 Blocks + EarlyStopping :

The next thing I did was take this approach, since it got the best accuracy so far, and added an early stopping technique which will stop the model at its best before it starts over fitting.
This resulted in an accuracy of 62% with no sign of over-fitting.  

## Future Work :

After that I looked at the confusing matrix to see which emotions the model was not doing well recognizing and which emotions the model was getting mixed up. The model was not recognizing the second emotion (Disgust), and had a problem confusing the emotion “sad” with (Netural, Fear, Angry), and “Neutral” with (Sad, Fear, Angry).

To further improve the model we would need bigger images with more features in them to train our model on.  I could also experiment with different model structures, and with the regulation techniques that did well such as DropOut, by applying it differently to different layers of the model, and trying out different rates. Exploring other regularization techniques such as Batch Normalization, Transfer learning and Pixel scaling.
