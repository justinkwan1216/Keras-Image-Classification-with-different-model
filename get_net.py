from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, GlobalMaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import InceptionResNetV2
from keras.applications import NASNetMobile
from keras.applications import MobileNetV2
from keras.optimizers import Adam, RMSprop, SGD
import efficientnet.keras as efn
def get_MobileNetV2(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = MobileNetV2(weights='imagenet', input_shape=IMAGE_SIZE, include_top=False)
    
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

    opt = Adam(lr=1e-5)
    #opt = SGD(lr=0.001, momentum=0.9)
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

def get_resnet50(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=IMAGE_SIZE)
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

    opt = Adam(lr=1e-3)
    #opt = SGD(lr=0.001, momentum=0.9)
    
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

def get_VGG16(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    #new_input = Input(shape=IMAGE_SIZE)
    net = VGG16(weights='imagenet', input_shape=IMAGE_SIZE, include_top=False)
    
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

    #opt = Adam(lr=1e-5)
    opt = optimizers.RMSprop(lr=2e-5)
    #opt = SGD(lr=0.001, momentum=0.9)
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

def get_efficientnet(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
    x = net.output
    #x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)

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
    
    #opt = Adam(lr=1e-3)
    opt = Adam(lr=1e-5)
    
    net_final.compile(loss="categorical_crossentropy",
                      optimizer=opt, metrics=['accuracy'])
    return net_final
def get_InceptionResNetV2(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.2):
    net = InceptionResNetV2(weights='imagenet', input_tensor=None, input_shape=IMAGE_SIZE,include_top=False)
    x = net.output
    x = Flatten()(x)


    # 增加 DropOut layer
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
         layer.trainable = True
           
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final
def get_NASNetMobile(IMAGE_SIZE,FREEZE_LAYERS,NUM_CLASSES,DROPOUT_RATE=0.5):
    input_tensor = Input(shape=IMAGE_SIZE)
    net = NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = net.output
    x = Flatten()(x)
    
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = BatchNormalization()(x)

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

    opt = Adam(lr=1e-3)
    #opt = SGD(lr=0.001, momentum=0.9)
    
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final

