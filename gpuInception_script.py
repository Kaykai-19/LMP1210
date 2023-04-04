import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
# Set the image size
    img_size = (224, 224)

    # Define data generators for training, validation, and testing sets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Define batch size and image size
    batch_size = 128

    # Define data directories
    train_dir = "/home/ikeade/projects/rrg-wanglab/ikeade/KOA_Severity_Data/train"
    val_dir = "/home/ikeade/projects/rrg-wanglab/ikeade/KOA_Severity_Data/val"
    test_dir = "/home/ikeade/projects/rrg-wanglab/ikeade/KOA_Severity_Data/test"

    # train_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/KOA_Severity_Data/train"
    # val_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/KOA_Severity_Data/val"
    # test_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/KOA_Severity_Data/test"

    # train_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/version_2.0/archive/train"
    # val_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/version_2.0/archive/val"
    # test_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/version_2.0/archive/test"


    # train_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/archive/train"
    # val_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/archive/val"
    # test_dir = "/Users/charles/Desktop/MS-1/LMP1210/Group_Project/archive/test"


    # Define data generators for training, validation, and testing sets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # upsample the files


    # Load the InceptionV3 model
    inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))

    # Freeze the layers in InceptionV3
    for layer in inceptionv3.layers:
        layer.trainable = False

    # Add a global spatial average pooling layer
    x = inceptionv3.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer
    #x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)


    # Add a classification layer
    predictions = Dense(4, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inceptionv3.input, outputs=predictions)

    # freeze some layers in the base model
    for layer in inceptionv3.layers[:50]:
        layer.trainable = False

    # unfreeze the rest of the layers
    for layer in inceptionv3.layers[50:]:
        layer.trainable = True

    # Define a learning rate scheduler function
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return 0.0001

    # Create a callback to update the learning rate after each epoch
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

    # Compile the model and train it
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(
        train_generator,
        epochs=2,
        validation_data=val_generator,
        verbose=1,
        callbacks=[lr_scheduler, early_stop]
    )

    model.summary()

    # Plot the training and validation Loss
    plt.ioff()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Loss.png")
    plt.close()

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Acc.png")
    plt.close()

    # Evaluate the model on the testing set
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print(f"Test accuracy: {test_acc}\n")

    # Calculate precision, recall, and F1-score
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)
    y_prob = model.predict(test_generator)

    y_true = test_generator.classes
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")


## Calculate ROC curve and AUC
    my_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    print(f"AUC: {my_auc}\n")

# Plot ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(test_generator.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']

    for i, color in zip(range(test_generator.num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
              ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve")
    plt.close()


    # Evaluate the model on the testing set
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print(f"Test accuracy: {test_acc}\n")


    save_model(model, 'inceptionv3.h5')
