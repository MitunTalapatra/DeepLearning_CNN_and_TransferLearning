# Deep Learning: CNN and TransferLearning for image classification

The aim of the project was to apply CNN and pre-trained model (MobileNet) for image classification. This project was conducted to be familiar with Deep Learning models.

## CNN:

A dataset of around 1200 images of different fingers was used for training the CNN model. According to the documentation of Spiced Academy, 'Convolutional Layers apply a sliding window that processes only a portion of the input at a time. They therefore represent spatial relationships between pixels. A Convolutional Layer contains not just one set of neurons for processing a portion of the input. They contain many sets that are called feature maps or filters. During training, each feature map specializes in a particular feature of the input.'

model parameters:
    
    # first conv and max pool layer
    Conv2D(filters=6,kernel_size=(3,3),strides=(1,1),padding='same',activation = 'relu',input_shape= (224, 224, 3)),
    MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    
    # second conv and max pool layer
    Conv2D(filters=16,kernel_size=(4,4),strides=(1,1),padding='same',activation = 'relu'),
    MaxPooling2D(pool_size=(4,4),strides=(2,2),padding='same'),
    
    # third conv and max pool layer
    Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='same',activation = 'relu'),
    MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    
    Flatten(),
    
    Dense(256,activation='relu'),
   
    Dense(128,activation='relu'),
    
    Dense(4,activation='softmax') # you could keep only this dense layer and remove the other dense layers
    ])
    
Accuracy (Train vs Validation)                  |  Loss (Train vs Validation)
:-------------------------:|:-------------------------:
<img width="407" alt="Screenshot 2022-04-26 at 22 05 20" src="https://user-images.githubusercontent.com/21356490/165383186-f46fe663-25ee-4e37-abee-3fe4bd5f4ded.png">  |  <img width="404" alt="Screenshot 2022-04-26 at 22 06 07" src="https://user-images.githubusercontent.com/21356490/165383222-8d1a4fba-f1fd-4e36-a0e7-0cdb035c5cbc.png">


## Transfer Learning:

To produce better result, pre-trained model mobilnetv2 was used as base model. MobilenetV2 is based on imgenet dataset. The base model was freezed not to train and later adding dense layers.

Here is the base model:

    base_model = mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    alpha=0.35,         # specific parameter of this model, small alpha reduces the number of overall weights
    pooling='avg',      # applies global average pooling to the output of the last conv layer (like a flattening)
    include_top=False,  # !!!!! we only want to have the base, not the final dense layers 
    input_shape=(224, 224, 3))
    
Freezing base model

    base_model.trainable = False

Final model parameters:

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(classes), activation='softmax')) #!!! Final layer with a length of 2, and softmax activation 
    # have a look at the trainable and non-trainable params statistic
    model.summary()

The model derived score of categorical accuracy of 0.9977 and validation accuracy of 0.9910.
