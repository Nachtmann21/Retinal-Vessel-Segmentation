import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)

# ——— 0) CONFIG —————————————————————————————————————————————
ROOT_DIR    = 'Drive Segmented'
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 50            # halved
AUTOTUNE    = tf.data.AUTOTUNE
VAL_STEPS   = 20            # small val split
TRAIN_STEPS = 100           # per-epoch random batches
PAIR_SAMPLE = 1000
TPR_TARGETS = [0.90, 0.95, 0.99]

# ——— 1) LIST PERSONS ———————————————————————————————————————————
persons = sorted([
    d for d in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, d))
])

# ——— 2) PAIR SAMPLER ———————————————————————————————————————————
def sample_pair(same: bool):
    if same:
        pid = random.choice(persons)
        a, b = random.sample(os.listdir(os.path.join(ROOT_DIR, pid)), 2)
        lbl = 1
        pA = os.path.join(ROOT_DIR, pid, a)
        pB = os.path.join(ROOT_DIR, pid, b)
    else:
        p1, p2 = random.sample(persons, 2)
        a = random.choice(os.listdir(os.path.join(ROOT_DIR, p1)))
        b = random.choice(os.listdir(os.path.join(ROOT_DIR, p2)))
        lbl = 0
        pA = os.path.join(ROOT_DIR, p1, a)
        pB = os.path.join(ROOT_DIR, p2, b)
    return pA, pB, lbl

# ——— 3) TF.DATA PAIR GENERATOR ————————————————————————————————————
def pair_generator():
    while True:
        same = (random.random() < 0.5)
        pA, pB, lbl = sample_pair(same)
        imgA = np.expand_dims(np.array(Image.open(pA).convert('L')), -1)
        imgB = np.expand_dims(np.array(Image.open(pB).convert('L')), -1)
        yield (imgA, imgB), np.int32(lbl)

output_signature = (
    (tf.TensorSpec((None, None, 1), tf.uint8),
     tf.TensorSpec((None, None, 1), tf.uint8)),
    tf.TensorSpec((), tf.int32)
)

ds = tf.data.Dataset.from_generator(
    pair_generator,
    output_signature=output_signature
)

# ——— 4) PREPROCESS & AUGMENT —————————————————————————————————————
def preprocess(pair, lbl):
    a, b = pair
    # uint8→float32 [0,1]
    a = tf.image.convert_image_dtype(a, tf.float32)
    b = tf.image.convert_image_dtype(b, tf.float32)
    # resize
    a = tf.image.resize(a, [IMG_SIZE, IMG_SIZE])
    b = tf.image.resize(b, [IMG_SIZE, IMG_SIZE])
    # random ±5° rotation + flip
    angle = tf.random.uniform([], -5, 5) * (np.pi / 180.0)
    a = tfa.image.rotate(a, angle, interpolation='BILINEAR')
    b = tfa.image.rotate(b, angle, interpolation='BILINEAR')
    a = tf.image.random_flip_left_right(a)
    b = tf.image.random_flip_left_right(b)
    return (a, b), tf.cast(lbl, tf.float32)

# build full batched dataset
full_ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE) \
            .batch(BATCH_SIZE) \
            .prefetch(AUTOTUNE)

val_ds        = full_ds.take(VAL_STEPS)
train_only_ds = full_ds.skip(VAL_STEPS)

# ——— 5) EMBEDDING + SIAMESE NET ————————————————————————————————————
def make_embedding_net():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return Model(inp, x, name='EmbedNet')

embed_net = make_embedding_net()

inA = layers.Input((IMG_SIZE, IMG_SIZE, 1))
inB = layers.Input((IMG_SIZE, IMG_SIZE, 1))
eA  = embed_net(inA)
eB  = embed_net(inB)

dist = layers.Lambda(
    lambda t: tf.sqrt(tf.reduce_sum(tf.square(t[0] - t[1]), axis=1, keepdims=True)),
    name='euclid_dist'
)([eA, eB])

siamese = Model([inA, inB], dist, name='SiameseNet')

# ——— 6) CONTRASTIVE LOSS & COMPILE ——————————————————————————————
def contrastive_loss(y_true, y_pred, margin=1.0):
    sq = tf.square(y_pred)
    md = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * sq + (1 - y_true) * md)

siamese.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=contrastive_loss
)

# ——— 7) CALLBACKS & TRAIN ——————————————————————————————————————
cb = [
    ModelCheckpoint(
        'best_siamese.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,                # stop earlier
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,                # quicker LR drop
        verbose=1
    )
]

siamese.summary()
history = siamese.fit(
    train_only_ds,
    steps_per_epoch=TRAIN_STEPS,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
    epochs=EPOCHS,
    callbacks=cb
)

# ——— 8) EVALUATE HELD-OUT PAIRS ——————————————————————————————————
siamese.load_weights('best_siamese.h5')

def get_test_pairs(n=PAIR_SAMPLE):
    dists, labs = [], []
    for _ in range(n):
        pA, pB, lbl = sample_pair(random.random() < 0.5)
        imgA = (np.expand_dims(np.array(Image.open(pA).convert('L')), -1) / 255.0)
        imgB = (np.expand_dims(np.array(Image.open(pB).convert('L')), -1) / 255.0)
        a = tf.image.resize(imgA, [IMG_SIZE, IMG_SIZE])[None]
        b = tf.image.resize(imgB, [IMG_SIZE, IMG_SIZE])[None]
        dists.append(siamese.predict([a, b], verbose=0)[0, 0])
        labs.append(lbl)
    return np.array(dists), np.array(labs)

dists, labs = get_test_pairs()
fpr, tpr, thr = roc_curve(labs, -dists)
roc_auc = auc(fpr, tpr)
print(f"\n>>> AUC = {roc_auc:.4f}\n")

# plot ROC
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], '--', c='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.show()

# thresholds & metrics
print(f"{'Method':<10} {'Thr':>6}  {'TPR':>5}  {'FPR':>5}  {'P':>5}  {'R':>5}  {'F1':>5}")
for tgt in TPR_TARGETS:
    idx   = np.argmax(tpr >= tgt)
    d0    = -thr[idx]
    ypred = (dists < d0).astype(int)
    prec  = precision_score(labs, ypred)
    rec   = recall_score(labs, ypred)
    f1    = f1_score(labs, ypred)
    tn, fp, fn, tp = confusion_matrix(labs, ypred, labels=[0,1]).ravel()
    print(f"TPR≥{int(tgt*100):<4}% {d0:6.3f}  {tp/(tp+fn)*100:5.1f}%  "
          f"{fp/(fp+tn)*100:5.1f}%  {prec*100:5.1f}%  {rec*100:5.1f}%  {f1*100:5.1f}%")

# show one confusion matrix
d0 = -thr[np.argmax(tpr >= TPR_TARGETS[0])]
cm = confusion_matrix(labs, (dists < d0).astype(int), labels=[0,1])
ConfusionMatrixDisplay(cm, display_labels=["DIFF","SAME"]) \
    .plot(cmap="Blues", values_format='d')
plt.title(f"Conf Matrix @ TPR≥{int(TPR_TARGETS[0]*100)}% (thr={d0:.3f})")
plt.show()
