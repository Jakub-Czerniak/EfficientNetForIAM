import sys

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import EfficientNetB0
import os
import matplotlib.pyplot as plt


# load data
words_list = []
np.random.seed(68)

words = open(f"words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if abs(int(line.split(" ")[3]) - int(line.split(" ")[4])) < 10:
        continue
    if abs(int(line.split(" ")[5]) - int(line.split(" ")[6])) < 10:
        continue
    if line.split(" ")[1] != "err":
        words_list.append(line)

words_count = len(words_list)
print('Words count: ' + str(words_count))

np.random.shuffle(words_list)

split_idx = int(words_count * 0.9)
val_split_idx = int(split_idx + (words_count - split_idx) * 0.5)
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:val_split_idx]
val_samples = words_list[val_split_idx:]
print('Splited into: ' + str(len(train_samples)) + ' trains samples, ' + str(len(test_samples)) + ' test samples, ' + str(len(val_samples)) + ' validation samples.')


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")
        image_name = line_split[0]
        folder_main_name = image_name.split("-")[0]
        folder_sub_name = image_name.split("-")[1]
        img_path = os.path.join('words', folder_main_name, folder_main_name + "-" + folder_sub_name, image_name + ".png")
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
val_img_paths, val_labels = get_image_paths_and_labels(val_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)


train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)
    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))

print('Max length ' + str(max_len))
print('Vocab size ' + str(len(characters)))
print('Example labels:')
print(train_labels_cleaned[:10])


def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


val_labels_cleaned = clean_labels(val_labels)
test_labels_cleaned = clean_labels(test_labels)

AUTOTUNE = tf.data.AUTOTUNE

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def resize_padding(image, height, width):
    image = tf.image.resize(image, size=(height, width), preserve_aspect_ratio=True)
    pad_height = height - tf.shape(image)[0]
    pad_width = width - tf.shape(image)[1]

    if pad_height % 2 != 0:
        pad_height_bottom = pad_height // 2
        pad_height_top = pad_height_bottom + 1
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        pad_width_bottom = pad_width // 2
        pad_width_top = pad_width_bottom + 1
    else:
        pad_width_top = pad_width_bottom = pad_width // 2

    image = tf.pad(image, paddings=((pad_height_top, pad_height_bottom), (pad_width_top, pad_width_bottom), (0, 0)))
    image = tf.transpose(image, perm=(1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image


batch_size = 64
padding_token = 99
image_width = 224
image_height = 224


def preprocess_image(image_path, img_height, img_width):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = resize_padding(image, img_height, img_width)
    image = tf.cast(image, tf.float32) / 255
    return image


def vectorize_label(label):
    print(label)
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def preprocess_image_label(image_path, label):
    image = preprocess_image(image_path, image_height, image_width)
    label = vectorize_label(label)
    return image, label


def dataset_prepare(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(preprocess_image_label, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


train_ds = dataset_prepare(train_img_paths, train_labels_cleaned)
val_ds = dataset_prepare(val_img_paths, val_labels_cleaned)
test_ds = dataset_prepare(test_img_paths, test_labels_cleaned)


def CTCLoss(y_true, y_pred):
    input_length = tf.math.count_nonzero(tf.cast(y_pred, tf.int64), dtype=tf.int64)
    label_length = tf.math.count_nonzero(tf.cast(y_true, tf.int64), dtype=tf.int64)
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# train model
# can try keras.losses.CategoricalCrossentropy() for loss
epochs = 10
model = EfficientNetB0(weights=None, classes=19, include_top=True, classifier_activation='softmax')
model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
history = model.fit(x=train_ds, epochs=epochs, validation_data=val_ds)
model.save('model.keras')


def plot_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_history(history)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_len, greedy=True)[0][0][:, max_len]
    output_text = []
    for result in results:
        result = tf.gather(result, tf.where(tf.math.not_equal(result, -1)))
        result = tf.strings.reduce_join(num_to_char(result)).numpy.decode("UTF-8")
        output_text.append(result)
    return output_text


for batch in test_ds.take(1):
    batch_images = batch['image']
    _, ax = plt.subplots(4, 4, figsize=(15, 8))
    preds = model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = 'Prediction: ' + pred_texts[i]
        ax[i // 4, i % 4].imshow(img, cmap='gray')
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis('off')

plt.show()

