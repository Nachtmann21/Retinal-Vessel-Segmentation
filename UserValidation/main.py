import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

root_dir = 'Drive Segmented'
# mixed_precision.set_global_policy('mixed_float16')
BATCH_SIZE = 16
NUM_EPOCHS = 200

class DeviceCheckCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        gpus = tf.config.list_logical_devices('GPU')
        if gpus:
            # Pick the first logical GPU
            print(f"✅  Training running on GPU: {gpus[0].name}")
        else:
            print("⚠️  No GPU found — training on CPU.")

def get_test_pairs(n=500):
    """
    Sample n random same/different pairs, run them through the Siamese model,
    and return two numpy arrays: distances and binary labels.
    """
    dists = []
    labs = []

    for _ in range(n):
        # 50/50 same vs diff
        same_flag = random.random() < 0.5
        imgA, imgB, lbl_str = sample_pair(same_flag)

        # PIL → numpy [0,1]
        arrA = np.array(imgA, dtype=np.float32) / 255.0
        arrB = np.array(imgB, dtype=np.float32) / 255.0

        # add channel dim → shape (H,W,1)
        arrA = np.expand_dims(arrA, axis=-1)
        arrB = np.expand_dims(arrB, axis=-1)

        # numpy → tf.Tensor, then resize to (IMG_SIZE,IMG_SIZE,1)
        tA = tf.convert_to_tensor(arrA)
        tB = tf.convert_to_tensor(arrB)
        tA = tf.image.resize(tA, [IMG_SIZE, IMG_SIZE])
        tB = tf.image.resize(tB, [IMG_SIZE, IMG_SIZE])

        # add batch axis → shape (1,IMG_SIZE,IMG_SIZE,1)
        batchA = tf.expand_dims(tA, 0)
        batchB = tf.expand_dims(tB, 0)

        # run through Siamese → get a scalar distance
        dist = siamese_model.predict([batchA, batchB], verbose=0)[0, 0]

        dists.append(dist)
        labs.append(1 if lbl_str == 'SAME' else 0)

    return np.array(dists), np.array(labs)

persons = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]
print("Detected persons:", persons)

def sample_pair(same: bool):
    if same:
        pid = random.choice(persons)
        imgs = os.listdir(os.path.join(root_dir, pid))
        a, b = random.sample(imgs, 2)
        label = 'SAME'
        p_a = os.path.join(root_dir, pid, a)
        p_b = os.path.join(root_dir, pid, b)
    else:
        p1, p2 = random.sample(persons, 2)
        a = random.choice(os.listdir(os.path.join(root_dir, p1)))
        b = random.choice(os.listdir(os.path.join(root_dir, p2)))
        label = 'DIFF'
        p_a = os.path.join(root_dir, p1, a)
        p_b = os.path.join(root_dir, p2, b)

    img_a = Image.open(p_a).convert('L')
    img_b = Image.open(p_b).convert('L')
    return img_a, img_b, label

samA, samB, _ = sample_pair(True)
difA, difB, _ = sample_pair(False)

fig, ax = plt.subplots(2, 2, figsize=(6,6))
ax[0,0].imshow(samA, cmap='gray'); ax[0,0].set_title('SAME : Left');  ax[0,0].axis('off')
ax[0,1].imshow(samB, cmap='gray'); ax[0,1].set_title('SAME : Right'); ax[0,1].axis('off')
ax[1,0].imshow(difA, cmap='gray'); ax[1,0].set_title('DIFF : Left');  ax[1,0].axis('off')
ax[1,1].imshow(difB, cmap='gray'); ax[1,1].set_title('DIFF : Right'); ax[1,1].axis('off')
plt.tight_layout()
plt.show()

# ——— Augmentations ———
rotated = samA.rotate(5, resample=Image.Resampling.BILINEAR)
flipped = samA.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

fig, ax2 = plt.subplots(1, 3, figsize=(9,3))
ax2[0].imshow(samA, cmap='gray');    ax2[0].set_title('Orig');    ax2[0].axis('off')
ax2[1].imshow(rotated, cmap='gray'); ax2[1].set_title('Rotated'); ax2[1].axis('off')
ax2[2].imshow(flipped, cmap='gray'); ax2[2].set_title('Flipped'); ax2[2].axis('off')
plt.tight_layout()
plt.show()

# —————————————————————————————————————————————————————————————————————————————————————————————————————————
# 1a) A tiny wrapper that yields raw NumPy arrays + int labels
def pair_generator():
    while True:
        # 50% same / 50% diff
        sam = random.random() < 0.5
        imgA_pil, imgB_pil, lbl_str = sample_pair(sam)
        # convert to H×W×1 uint8 arrays
        a = np.expand_dims(np.array(imgA_pil, dtype=np.uint8), -1)
        b = np.expand_dims(np.array(imgB_pil, dtype=np.uint8), -1)
        # label: SAME→1, DIFF→0
        lbl = np.int32(1 if lbl_str == 'SAME' else 0)
        yield (a, b), lbl

# 1b) Define the Dataset from that generator
output_signature = (
    (tf.TensorSpec(shape=(None,None,1), dtype=tf.uint8),
     tf.TensorSpec(shape=(None,None,1), dtype=tf.uint8)),
    tf.TensorSpec(shape=(), dtype=tf.int32)
)

ds = tf.data.Dataset.from_generator(
    pair_generator,
    output_signature=output_signature
)

# —————————————————————————————————————————————————————————————————————————————————————————————————————————
# 2a) constants
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE

# 2b) a map function
def preprocess_and_augment(pair, label):
    imgA, imgB = pair

    # decode/rescale from [0,255]→[0,1]
    imgA = tf.image.convert_image_dtype(imgA, tf.float32)
    imgB = tf.image.convert_image_dtype(imgB, tf.float32)

    # resize
    imgA = tf.image.resize(imgA, [IMG_SIZE, IMG_SIZE])
    imgB = tf.image.resize(imgB, [IMG_SIZE, IMG_SIZE])

    # random small rotation (±5°) & flip
    # Note: tf.keras.preprocessing.image.random_rotation is slow,
    # so here’s a quick manual rotate by a random angle.
    angle = tf.random.uniform([], -5, 5) * (3.14159/180.0)

    # Rotations with --pip install tensorflow-addons
    # imgA = tfa.image.rotate(imgA, angle, interpolation='BILINEAR')
    # imgB = tfa.image.rotate(imgB, angle, interpolation='BILINEAR')

    imgA = tf.image.random_flip_left_right(imgA)
    imgB = tf.image.random_flip_left_right(imgB)

    return (imgA, imgB), tf.cast(label, tf.float32)

# 2c) build the final pipeline
train_ds = (
    ds
    .map(preprocess_and_augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

for (batchA, batchB), batchY in train_ds.take(1):
    print("A batch shape:", batchA.shape)   # ⇒ (32,224,224,1)
    print("B batch shape:", batchB.shape)   # ⇒ (32,224,224,1)
    print("Labels   :", batchY.numpy()[:10])# ten 0/1 values
    break

# —————————————————————————————————————————————————————————————————————————————————————————————————————————
# 3a) Define the embedding (base) network - Model definition
def make_embedding_model(input_shape=(224,224,1), embedding_dim=128):
    inp = layers.Input(input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    # L2‐normalize to lie on the unit hypersphere
    x = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return Model(inp, x, name="EmbeddingNet")

# 3b) Instantiate it
embed_net = make_embedding_model()
embed_net.summary()

# 3c) Build the Siamese distance model
# Two image‐inputs
input_a = layers.Input((224,224,1), name="imgA")
input_b = layers.Input((224,224,1), name="imgB")

# Shared embedding
emb_a = embed_net(input_a)
emb_b = embed_net(input_b)

# Euclidean distance: ||emb_a – emb_b||₂
distance = layers.Lambda(
    lambda tensors: tf.sqrt(
        tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)
    ),
    name="euclid_dist"
)([emb_a, emb_b])

siamese_model = Model([input_a, input_b], distance, name="SiameseNet")
siamese_model.summary()

# 3d) Contrastive loss
def contrastive_loss(y_true, y_pred, margin=1.0):
    # y_true ∈ {0,1}, y_pred = distance ≥ 0
    squared = tf.square(y_pred)
    margin_dist = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * squared + (1 - y_true) * margin_dist)

# 4) Compile the model ———————————————————————————————————————————————————————————————————————————————————————————————
siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=contrastive_loss
)

# val split of 50 batches:
val_ds = train_ds.take(50)
train_only_ds = train_ds.skip(50)

# 5) Train the model ———————————————————————————————————————————————————————————————————————————————————————————————
# Callbacks
callbacks = [
    DeviceCheckCallback(),
    ModelCheckpoint(
        'best_siamese.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    TensorBoard(log_dir='logs/siamese', histogram_freq=1)
]

# *********** TRAINING ***********
history = siamese_model.fit(
    train_only_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch = 50,
    # steps_per_epoch = total_pairs // BATCH_SIZE
    validation_data=val_ds,
    validation_steps=50,
    callbacks=callbacks
)

# ——— 6) Plot train/val loss ———
plt.plot(history.history['loss'],      label='train loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Contrastive Loss')
plt.legend()
plt.show()

# ——— 7) ROC & Threshold Selection ———
dists, labs = get_test_pairs(500)
fpr, tpr, thresholds = roc_curve(labs, -dists)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

idx = np.argmax(tpr >= 0.95)
score_thr = thresholds[idx]   # threshold on -dists
dist_thr  = -score_thr        # convert back to threshold on dists
print(f"AUC = {roc_auc:.3f}")
print(f"Distance threshold for 95% TPR: {dist_thr:.3f}")

# ——— 8) Save your trained models ———
siamese_model.save("siamese_model.h5")
embed_net.save("embed_net.h5")
print("✅ Saved siamese_model.h5 and embed_net.h5")

# ——— 9) Confusion matrix at your chosen threshold ———
y_pred = (dists < dist_thr).astype(int)
cm = confusion_matrix(labs, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["DIFF","SAME"])
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Confusion Matrix @ dist_thr={dist_thr:.3f}")
plt.show()

