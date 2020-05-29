from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalMaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2, Xception, InceptionV3
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
#EfficientNetB0
#import efficientnet.keras as efn
def get_InceptionV3(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = InceptionV3(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    
    x = net.output
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    #x = Dense(1024, activation='relu')(x)
    
    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    x = BatchNormalization()(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
            
    return net_final

def get_Xception(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = Xception(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
            
    return net_final

def get_MobileNetV2(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = MobileNetV2(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
            
    return net_final

def get_resnet50(weights,IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = ResNet50(weights=weights, include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    x = net.output

    #x = GlobalAveragePooling2D()(x)
    #x = Dense(256, activation='relu')(x)
    x = Flatten()(x)

    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    #x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True

    return net_final

def get_VGG16(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    #new_input = Input(shape=IMAGE_SIZE)
    net = VGG16(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True

    return net_final

def get_efficientnetB0(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    x = net.output
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    # net_final.add(layers.Flatten(name="flatten"))
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)
    # net_final.add(layers.Dense(256, activation='relu', name="fc1"))

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    #x = Dense(128, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    
    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    return net_final
def get_InceptionResNetV2(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=None, input_shape=IMAGE_SIZE)
    x = net.output
    x = Flatten()(x)


    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    #x = Dense(128, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
  
    return net_final
def get_NASNetLarge(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    input_tensor = Input(shape=IMAGE_SIZE)
    
    net = NASNetLarge(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = net.output
    x = Flatten()(x)
    
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    
    #x = BatchNormalization()(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dropout(0.5)(x)
    
    #x = BatchNormalization()(x)

    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    #x = Dense(128, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    return net_final

