# SETL_Pothole: Stacking Ensemble Learning
# Xception + EfficientNetB0 + Logistic Regression Meta Learner

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
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
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
val_data = val_datagen.flow_from_directory(
    "dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
test_data = val_datagen.flow_from_directory(
    "dataset/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

input_tensor = Input(shape=(224, 224, 3))

xception_base = Xception(weights="imagenet", include_top=False, input_tensor=input_tensor)
x1 = GlobalAveragePooling2D()(xception_base.output)

efficient_base = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=input_tensor)
x2 = GlobalAveragePooling2D()(efficient_base.output)

merged = Concatenate()([x1, x2])
x = Dense(512, activation="relu")(merged)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

base_model = Model(inputs=input_tensor, outputs=output)

base_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history = base_model.fit(
    train_data, validation_data=val_data, epochs=EPOCHS
)

train_data.reset()
val_data.reset()
test_data.reset()

train_pred = base_model.predict(train_data, verbose=1).ravel()
test_pred = base_model.predict(test_data, verbose=1).ravel()

X_train_stack = train_pred.reshape(-1, 1)
X_test_stack = test_pred.reshape(-1, 1)

y_train = train_data.classes
y_test = test_data.classes

meta_classifier = LogisticRegression()
meta_classifier.fit(X_train_stack, y_train)

stack_probs = meta_classifier.predict_proba(X_test_stack)[:, 1]
stack_pred = (stack_probs >= 0.5).astype(int)

print("\n================ SETL_Pothole ================")
print("Accuracy:", accuracy_score(y_test, stack_pred))
print("AUC:", roc_auc_score(y_test, stack_probs))
print("\nClassification Report:")
print(classification_report(y_test, stack_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, stack_pred))

fpr, tpr, _ = roc_curve(y_test, stack_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"SETL AUC = {roc_auc_score(y_test, stack_probs):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SETL_Pothole")
plt.legend()
plt.grid()
plt.show()
