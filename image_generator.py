from keras.preprocessing.image import ImageDataGenerator
# 透過 data augmentation 產生訓練與驗證用的影像資料
def get_generator(TRAIN_DATA_DIR,IMAGE_SIZE,BATCH_SIZE,r_r,w_s_r,h_s_r,s_r,z_r,c_s_r,h_f,v_f,FILL_MODE,VALIDATION_SPLIT):
    train_datagen = ImageDataGenerator(rotation_range=r_r,
                                   width_shift_range=w_s_r,
                                   height_shift_range=h_s_r,
                                   shear_range=s_r,
                                   zoom_range=z_r,
                                   channel_shift_range=c_s_r,
                                   horizontal_flip=h_f,
                                   vertical_flip=v_f,
                                   fill_mode=FILL_MODE,
                                   validation_split=VALIDATION_SPLIT)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training') # set as training data

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation') # set as training data

    # 輸出各類別的索引值
    for cls, idx in train_generator.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))

    return train_generator,validation_generator
