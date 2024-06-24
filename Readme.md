# Objective
To build an Neural Network architecture for semantic segmentation on the cityscapes dataset 

# Introduction
Semantic segmentation is  a computer vision task where the goal is to classify each pixel in an image into a predefined category or class. Unlike traditional image classification, which assigns a single label to an entire image, semantic segmentation provides a detailed understanding of the scene by labelling each pixel individually. 
Semantic segmentation architectures usually consist of  an encoder and a decoder,
wherein the encoder is supposed to downsample the image and the decoder upsample,
And there are skip connections feeding the downsampled feature map to the decoder net at each level.
This is a strong approach but lacks a number of things, for example the ability to capture context at different spatial levels/contexts 
And so we explore a number of methods in our project in order to get better predictions and build on existing architectures.

# Thought Process
We knew we had to go for an encoder/decoder type architecture with skip connections in between and our first approach consisted of residual connections/skip connections connecting the features of previous feature maps with newly created ones within the separate encoder and decoder nets, and the benefit of this was that it would facilitate information propagation leading to better performance with lesser number of parameters.

This idea achieved great relative success. With a meagre 9 million parameters, the model showed excellent accuracy (83 percent) however a poor MeanIOU(25 percent) on validation data.
Model was able to identify common structures such as cars, roads and buildings but failed to capture minute details such as warning signs, lamp posts, etc 
However, the model failed to get more than 87% training accuracy which is indicative of high bias in the model.

Although this was a good approach it still didnt fix the problem of having a constrained/ restrictive/ fixed receptive field, and so taking inspiration of Inception Net  we concatenated not only singular filter size outputs, but outputs of  1x1, 3x3 and 5x5 convolutions,

1x1 filters: operate on individual pixels or small local neighbourhoods without considering much surrounding context.
3x3 filters: capture features within a moderate receptive field, considering both local details and some degree of surrounding context.
5x5 filters: have a larger receptive field, allowing them to capture features over a broader spatial range and incorporate more global context.

We were also thinking to combine the strength of  all pertain models into a single model , yet we know that it is the complex task but we wanted to model to have -
1. VGG for its simplicity and effectiveness in feature extraction.
2. UNet for its encoder-decoder structure with skip connections.
3. ResNet for its deep residual learning capabilities.
4. Inception for its multi-scale feature extraction.
We aimed to develop a hybrid model that amalgamates the distinctive qualities of various architectures into a unified framework.

# Approaches
First Approach was to build a UNET type architecture with residual connections within the encoder and decoder and train it. 
Later, we decided to use the pretrained inception network and ResNet50  itself as encoder layer however, it showed poor performance. Which was expected because inception network and Resnet are built for image classification not segmentation.

Another approach was to use VGG-16 as encoder and Unet's decoder as decoder with Convolved skip connections. However, this approach also failed to give better results. 

We also explored an approach that combines FCN  and Mask R-CNN. But it failed due to the complexity of integrating these architectures and the resultant compatibility issues, leading to suboptimal performance.

Finally we decided to add Inception Net type convolutional nets within the encoder and decoder, essentially building on our first (residual) approach


# Working Steps
## Preprocessing
Images from the dataset are batched in size of 16 mini-batches using tensorflow.data.Dataset API. Each image is then resized to 128 x 256 pixels using InterArea Interpolation method. Each image is then normalised by dividing by 255. 

Cityscapes Dataset has a label image for each training image which has integer labels corresponding to every pixel in the training image. Label image is also resized to 128x256 but, by using Nearest Neighbour Resampling method so that labels remain integers after resizing. 

## Network Architecture
Network Architecture consists of  three main components : Stack Module, Reduction Module,Upsampling, Skip Connection, Softmax Output

### 1. Stack Module
Stack consists of 3x3 convolution followed by another 3x3 convolution of input 
concatenated with another 5x5 convolution of input

### 2. Reduction Module
Due to concatenation of layers in the stack module, the resultant output has an arbitrarily large number of channels. To reduce the no. of channels, a reduction module is present after every stack module. Reduction module is essentially a 1x1 convolution layer with n filters where n in our model in the no. of filters of each layer in stack module.

### 3. Upsampling Layers
After the decoder layer, the values are upsampled to twice the size. Value is repeated when upsampling

### 4. Skip Connections
Skip connections are provided between the encoder and the decoder layer

### 5. Softmax Outputs
Softmax Output is basically an 1x1 convolution with 34 filters whose softmax is evaluated

## Loss Function
Sparse Categorical Cross entropy was chosen as a loss function because we are using integer labels to train the model. 

# Conclusion
We are able to successfully construct multiple models for semantic segmentation which  are at par with well known architectures 


In conclusion, our journey to develop a robust neural network for image semantic segmentation led us through various architectures like UNet, VGG, ResNet, FCN, and Inception. Despite encountering challenges such as memory constraints and versioning conflicts, we remained determined to create a model that combined the best features of these architectures.
Ultimately, we tried out a number of architectures from which three stood out
    1. SegCeption net: an approach similar to the Residual U-Net approach
    2. VGG16-UNet approach
    3. Residual Inception-UNet

Although our quest for the perfect model continues, we've gained invaluable insights and are optimistic about our future refinements.

# Future Prospects
To enhance our semantic segmentation model, we plan to explore several promising avenues:
    1. Advanced Pretrained Models: Use EfficientNet or Swin Transformer as encoders for better feature extraction and accuracy.
    2. Attention Mechanisms: Integrate attention mechanisms (e.g., SE blocks, self-attention) to focus on relevant image parts and improve segmentation detail.
    3.Data Augmentation: Apply advanced data augmentation and synthetic data generation to create a diverse training dataset and improve generalization
    4. Custom Loss Functions: Experiment with custom loss functions such as dice loss and mean IoU to improve model performance on specific segmentation challenges
    
These strategies will help us build more robust, accurate, and efficient segmentation models, leveraging the latest deep learning advancements.


# Links
Notebook Link : https://www.kaggle.com/code/ishansrivastava1308/semanticsegmentationcityscapes

Repository Link : https://github.com/Ishan130803/Semantic-Segmentation-Kaggle





