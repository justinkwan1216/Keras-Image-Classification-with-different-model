from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, GlobalMaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import InceptionResNetV2
from keras.optimizers import Adam, RMSprop
import efficientnet.keras as efn
def get_resnet50(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=IMAGE_SIZE)
    x = net.output
    x = Flatten()(x)

 
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)


    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)


    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
           

    net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

def get_VGG16(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):

    net = VGG16(weights='imagenet', input_shape=IMAGE_SIZE, include_top=False)
    
    x = net.output
    x = Flatten()(x)


    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)


    x = Dense(256, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)


    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
           

    net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

def get_efficientnet(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
    x = net.output
    x = GlobalMaxPooling2D(name="gap")(x)

    # net_final.add(layers.Flatten(name="flatten"))
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)
    # net_final.add(layers.Dense(256, activation='relu', name="fc1"))


    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    

    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
    
    net_final.compile(loss="categorical_crossentropy",
                      optimizer=RMSprop(lr=2e-5),metrics=["acc"])
    return net_final
def get_InceptionResNetV2(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = InceptionResNetV2(weights='imagenet', input_tensor=None, input_shape=IMAGE_SIZE,include_top=False)
    x = net.output
    x = Flatten()(x)



    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)


    x = Dense(128, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)


    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
           

    net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

    
