

# ================= SETL_Pothole.py =================
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
    train_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    rescale=1./255
)
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


# ================= VETL_Pothole.py =================
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
BATCH_SIZE = 16
EPOCHS = 50

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True
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


# ================= BETL_Pothole.py =================
# BETL_Pothole: Bagging Ensemble Learning
# Bootstrap sampling + multiple CNN learners

import os
import shutil
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
BATCH_SIZE = 16
EPOCHS = 30
NUM_BAG_MODELS = 5

SOURCE_TRAIN = "dataset/train"
BAGGING_ROOT = "dataset/bagging"

classes = [
    d for d in os.listdir(SOURCE_TRAIN)
    if os.path.isdir(os.path.join(SOURCE_TRAIN, d))
]

os.makedirs(BAGGING_ROOT, exist_ok=True)

# Create bootstrap datasets
for bag_id in range(NUM_BAG_MODELS):
    bag_path = os.path.join(BAGGING_ROOT, f"bag_{bag_id}")
    os.makedirs(bag_path, exist_ok=True)

    for class_name in classes:
        source_class = os.path.join(SOURCE_TRAIN, class_name)
        target_class = os.path.join(bag_path, class_name)
        os.makedirs(target_class, exist_ok=True)

        images = [
            f for f in os.listdir(source_class)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        sampled_images = np.random.choice(
            images, size=len(images), replace=True
        )

        for i, image_name in enumerate(sampled_images):
            source_file = os.path.join(source_class, image_name)
            target_file = os.path.join(target_class, f"{i}_{image_name}")
            shutil.copy2(source_file, target_file)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    "dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
test_data = test_datagen.flow_from_directory(
    "dataset/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

def create_bagging_model(model_type):
    inputs = Input(shape=(224, 224, 3))

    if model_type == "xception":
        base = Xception(weights="imagenet", include_top=False, input_tensor=inputs)
    else:
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

bagging_models = []

for bag_id in range(NUM_BAG_MODELS):
    print(f"\nTraining Bagging Model {bag_id + 1}")

    bag_path = os.path.join(BAGGING_ROOT, f"bag_{bag_id}")

    bag_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    bag_data = bag_datagen.flow_from_directory(
        bag_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True
    )

    model_type = "xception" if bag_id % 2 == 0 else "efficientnet"
    model = create_bagging_model(model_type)

    model.fit(
        bag_data, validation_data=val_data,
        epochs=EPOCHS, verbose=1
    )

    bagging_models.append(model)

bag_predictions = []

for i, model in enumerate(bagging_models):
    print(f"Predicting with Bagging Model {i + 1}")
    test_data.reset()
    probabilities = model.predict(test_data, verbose=0).ravel()
    bag_predictions.append(probabilities)

bag_predictions = np.array(bag_predictions)

bagging_probs = np.mean(bag_predictions, axis=0)
bagging_pred = (bagging_probs >= 0.5).astype(int)

y_true = test_data.classes

print("\n================ BETL_Pothole ================")
print("Accuracy:", accuracy_score(y_true, bagging_pred))
print("AUC:", roc_auc_score(y_true, bagging_probs))
print("\nClassification Report:")
print(classification_report(y_true, bagging_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, bagging_pred))

fpr, tpr, _ = roc_curve(y_true, bagging_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"BETL AUC = {roc_auc_score(y_true, bagging_probs):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - BETL_Pothole")
plt.legend()
plt.grid()
plt.show()
