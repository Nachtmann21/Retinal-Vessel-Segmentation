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
from sklearn.model_selection import train_test_split

# ——— 0) CONFIG —————————————————————————————————————————————
ROOT_DIR    = 'Drive Segmented'
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS       = 50
AUTOTUNE     = tf.data.AUTOTUNE
PAIR_SAMPLE  = 1000
TPR_TARGETS  = [0.90, 0.95, 0.99]

# ——— 1) PERSON‐WISE SPLIT (3/1/1) ——————————————————————————
persons = sorted(d for d in os.listdir(ROOT_DIR)
                 if os.path.isdir(os.path.join(ROOT_DIR, d)))
splits = {}
for p in persons:
    imgs = [os.path.join(ROOT_DIR, p, f)
            for f in os.listdir(os.path.join(ROOT_DIR, p))
            if f.lower().endswith('.png')]
    # first remove 1 test
    train_val, test = train_test_split(imgs, test_size=1, random_state=42)
    # then remove 1 val
    train, val = train_test_split(train_val, test_size=1, random_state=42)
    splits[p] = {'train': train, 'val': val, 'test': test}

# ——— 2) PAIR SAMPLER BY SUBSET ——————————————————————————————————
def sample_pair(subset: str):
    same = (random.random() < 0.5)
    if same:
        p = random.choice(persons)
        a, b = random.sample(splits[p][subset], 2)
        label = 1
    else:
        p1, p2 = random.sample(persons, 2)
        a = random.choice(splits[p1][subset])
        b = random.choice(splits[p2][subset])
        label = 0
    return a, b, label

# ——— 3) DATASET FACTORY —————————————————————————————————————
def make_pair_ds(subset: str):
    def gen():
        while True:
            a_fp, b_fp, lbl = sample_pair(subset)
            yield (a_fp.encode(), b_fp.encode()), lbl

    output_sig = (
        (tf.TensorSpec([], tf.string),
         tf.TensorSpec([], tf.string)),
        tf.TensorSpec([], tf.int32)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)

    def load_and_preprocess(pair, lbl):
        fa, fb = pair
        ia = tf.io.decode_png(tf.io.read_file(fa), channels=1)
        ib = tf.io.decode_png(tf.io.read_file(fb), channels=1)
        ia = tf.image.convert_image_dtype(ia, tf.float32)
        ib = tf.image.convert_image_dtype(ib, tf.float32)
        ia = tf.image.resize(ia, [IMG_SIZE, IMG_SIZE])
        ib = tf.image.resize(ib, [IMG_SIZE, IMG_SIZE])
        if subset == 'train':
            # ±5° rotation + flip
            angle = tf.random.uniform([], -5, 5) * (np.pi/180.0)
            ia = tfa.image.rotate(ia, angle, interpolation='BILINEAR')
            ib = tfa.image.rotate(ib, angle, interpolation='BILINEAR')
            ia = tf.image.random_flip_left_right(ia)
            ib = tf.image.random_flip_left_right(ib)
        return (ia, ib), tf.cast(lbl, tf.float32)

    return (ds
            .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

train_ds = make_pair_ds('train')
val_ds   = make_pair_ds('val')
test_ds  = make_pair_ds('test')

# ——— 4) BUILD SIAMESE MODEL ————————————————————————————————————
def make_embedding_net():
    inp = layers.Input((IMG_SIZE,IMG_SIZE,1))
    x = layers.Conv2D(32,3,padding='same',activation='relu')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.Lambda(lambda v: tf.math.l2_normalize(v,axis=1))(x)
    return Model(inp, x, name='EmbedNet')

embed_net = make_embedding_net()
inA = layers.Input((IMG_SIZE,IMG_SIZE,1)); inB = layers.Input((IMG_SIZE,IMG_SIZE,1))
eA = embed_net(inA); eB = embed_net(inB)
dist = layers.Lambda(lambda t: tf.sqrt(
    tf.reduce_sum(tf.square(t[0]-t[1]),axis=1,keepdims=True)
), name='euclid_dist')([eA,eB])
siamese = Model([inA,inB], dist, name='SiameseNet')

def contrastive_loss(y_true,y_pred,margin=1.0):
    sq = tf.square(y_pred)
    md = tf.square(tf.maximum(margin - y_pred,0))
    return tf.reduce_mean(y_true*sq + (1-y_true)*md)

siamese.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=contrastive_loss
)

# ——— 5) TRAIN ——————————————————————————————————————————————
cb = [
    ModelCheckpoint('best_siamese.h5', save_best_only=True,
                    monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=2, verbose=1)
]

siamese.summary()
history = siamese.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb
)

# plot losses
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

# ——— 6) EVAL ON “TEST” PAIRS —————————————————————————————————
siamese.load_weights('best_siamese.h5')

def get_dists_labels(n=PAIR_SAMPLE):
    dists, labs = [], []
    for _ in range(n):
        a_fp,b_fp,lbl = sample_pair('test')
        ia = np.expand_dims(np.array(Image.open(a_fp).convert('L')), -1)/255.
        ib = np.expand_dims(np.array(Image.open(b_fp).convert('L')), -1)/255.
        a = tf.image.resize(ia, [IMG_SIZE,IMG_SIZE])[None]
        b = tf.image.resize(ib, [IMG_SIZE,IMG_SIZE])[None]
        dists.append(siamese.predict([a,b],verbose=0)[0,0])
        labs.append(lbl)
    return np.array(dists), np.array(labs)

dists, labs = get_dists_labels()
fpr, tpr, thr = roc_curve(labs, -dists)
roc_auc = auc(fpr,tpr)
print(f"\n>>> TEST‐AUC = {roc_auc:.4f}\n")

plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--',c='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.show()

print(f"{'Method':<8}{'Thr':>6}{'TPR':>8}{'FPR':>8}{'P':>8}{'R':>8}{'F1':>8}")
for tgt in TPR_TARGETS:
    idx = np.argmax(tpr>=tgt)
    d0  = -thr[idx]
    ypred = (dists<d0).astype(int)
    prec = precision_score(labs, ypred)
    rec  = recall_score(labs, ypred)
    f1   = f1_score(labs, ypred)
    tn, fp, fn, tp = confusion_matrix(labs,ypred,labels=[0,1]).ravel()
    print(f"TPR≥{int(tgt*100):<4}%{d0:8.3f}"
          f"{tp/(tp+fn)*100:8.1f}%"
          f"{fp/(fp+tn)*100:8.1f}%"
          f"{prec*100:8.1f}%"
          f"{rec*100:8.1f}%"
          f"{f1*100:8.1f}%")

# single confusion matrix at first TPR target
d0 = -thr[np.argmax(tpr>=TPR_TARGETS[0])]
cm = confusion_matrix(labs,(dists<d0).astype(int),labels=[0,1])
ConfusionMatrixDisplay(cm,display_labels=["DIFF","SAME"])\
  .plot(cmap="Blues",values_format='d')
plt.title(f"Confusion @ TPR≥{int(TPR_TARGETS[0]*100)}% (thr={d0:.3f})")
plt.show()
