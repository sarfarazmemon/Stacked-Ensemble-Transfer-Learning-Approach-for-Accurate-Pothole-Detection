# VETL_Pothole: Voting Ensemble Learning
# Xception + EfficientNetB0 using soft voting

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3
L2_REG = 1e-4

train_datagen = ImageDataGenerator(
    rotation_range=15,
            horizontal_flip=True,
            width_shift_range=0.10,
            height_shift_range=0.10,
            zoom_range=0.10,
            rescale=1./255
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=True
)
val_data = test_datagen.flow_from_directory(
    "dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
test_data = test_datagen.flow_from_directory(
    "dataset/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

def create_xception_model():
    inputs = Input(shape=(224, 224, 3))
    base = Xception(weights="imagenet", include_top=False, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def create_efficientnet_model():
    inputs = Input(shape=(224, 224, 3))
    base = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

xception_model = create_xception_model()
efficientnet_model = create_efficientnet_model()

xception_model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
efficientnet_model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

test_data.reset()
xception_probs = xception_model.predict(test_data, verbose=1).ravel()

test_data.reset()
efficientnet_probs = efficientnet_model.predict(test_data, verbose=1).ravel()

voting_probs = (xception_probs + efficientnet_probs) / 2
voting_pred = (voting_probs >= 0.5).astype(int)

y_true = test_data.classes

print("\n================ VETL_Pothole ================")
print("Accuracy:", accuracy_score(y_true, voting_pred))
print("AUC:", roc_auc_score(y_true, voting_probs))
print("\nClassification Report:")
print(classification_report(y_true, voting_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, voting_pred))

fpr, tpr, _ = roc_curve(y_true, voting_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"VETL AUC = {roc_auc_score(y_true, voting_probs):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - VETL_Pothole")
plt.legend()
plt.grid()
plt.show()
