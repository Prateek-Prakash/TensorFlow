#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow Version
print(f'TensorFlow: {tf.__version__}')

# Define Epochs
EPOCHS = 50

# IMPORT: Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# EXPLORE: Show Random Training Image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# PREPROCESS: Scale Images
train_images = train_images / 255.0
test_images = test_images / 255.0

# PREPROCESS: Verify Some Data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# BUILD: Setup Layers
model = tf.keras.Sequential([
    # Reformat (2D Array -> 1D Array)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Neurons (Learning)
    tf.keras.layers.Dense(128, activation='relu'),
    # Output Nodes (Logits)
    tf.keras.layers.Dense(10)
])

# BUILD: Compile Model
# Loss -- Minimize This
# Optimizer -- How Model Updates (Based Off Loss)
# Metrics -- Monitor Training & Testing Steps
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# TRAIN: Feed Model
model.fit(train_images, train_labels, epochs=EPOCHS)

# TRAIN: Evaluate Accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test Accuracy: {test_acc}')

# TRAIN: Make Predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# Class Probablities (Each Array Size = 10)
predictions = probability_model.predict(test_images)
# Check First Prediction
guess_label = np.argmax(predictions[0])
guess_class = class_names[guess_label]
test_label = test_labels[0]
test_class = class_names[test_label]
print(f'Guess Label: {guess_class}')
print(f'Test Label: {test_class}')

# HELPER METHOD: Plot Image
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
# HELPER METHOD: Plot Value Array
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# TRAIN: Plot Prediction (Correct)
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# TRAIN: Plot Prediction (Incorrect)
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# TRAIN: Plot N-Predictions
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# PREDICT: Use Trained Model
# Data Setup
img = test_images[1]
img = (np.expand_dims(img, 0))
# Predict Class
predictions_single = probability_model.predict(img)
# Plot Prediction
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
