import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# ——— 0. Setup —————————————————————————————————————————————————————
root_dir = 'Drive Segmented'
IMG_SIZE = 224

# Tell TF how to reconstruct your custom loss when loading
def contrastive_loss(y_true, y_pred, margin=1.0):
    squared    = tf.square(y_pred)
    margin_dist= tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * squared + (1 - y_true) * margin_dist)

# ——— 1. Load your best‐checkpointed Siamese & embedding nets —————————————
siamese_model = tf.keras.models.load_model(
    'best_siamese.h5',
    custom_objects={'contrastive_loss': contrastive_loss}
)
embed_net = tf.keras.models.load_model('embed_net.h5')

# ——— 2. Helpers for sampling & evaluation ————————————————————————————
persons = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]

def sample_pair(same: bool):
    # exactly the same sampler you used during training
    if same:
        pid = random.choice(persons)
        imgs = os.listdir(os.path.join(root_dir, pid))
        a, b = random.sample(imgs, 2)
        pA = os.path.join(root_dir, pid, a)
        pB = os.path.join(root_dir, pid, b)
        lbl = 1
    else:
        p1, p2 = random.sample(persons, 2)
        a = random.choice(os.listdir(os.path.join(root_dir, p1)))
        b = random.choice(os.listdir(os.path.join(root_dir, p2)))
        pA = os.path.join(root_dir, p1, a)
        pB = os.path.join(root_dir, p2, b)
        lbl = 0
    imgA = Image.open(pA).convert('L')
    imgB = Image.open(pB).convert('L')
    return imgA, imgB, lbl

def get_test_pairs(n=500):
    dists, labs = [], []
    for _ in range(n):
        imgA, imgB, lbl = sample_pair(random.random() < 0.5)
        # PIL→numpy→[0,1], add channel dim
        arrA = np.expand_dims(np.array(imgA, np.float32)/255., -1)
        arrB = np.expand_dims(np.array(imgB, np.float32)/255., -1)
        # numpy→tensor→resize
        tA = tf.image.resize(tf.convert_to_tensor(arrA), [IMG_SIZE,IMG_SIZE])
        tB = tf.image.resize(tf.convert_to_tensor(arrB), [IMG_SIZE,IMG_SIZE])
        # batch axis
        batchA = tf.expand_dims(tA,0)
        batchB = tf.expand_dims(tB,0)
        # predict distance
        dist = siamese_model.predict([batchA, batchB], verbose=0)[0,0]
        dists.append(dist)
        labs.append(lbl)
    return np.array(dists), np.array(labs)

# ——— 3. Compute ROC & pick threshold ———————————————————————————————
dists, labs = get_test_pairs(1000)   # increase for stability
fpr, tpr, thresholds = roc_curve(labs, -dists)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# threshold at 95% TPR
idx = np.argmax(tpr >= 0.95)
score_thr = thresholds[idx]
dist_thr  = -score_thr
print(f"AUC          : {roc_auc:.3f}")
print(f"Dist threshold @95% TPR: {dist_thr:.3f}")

# ——— 4. Confusion Matrix ———————————————————————————————————————————
y_pred = (dists < dist_thr).astype(int)
cm = confusion_matrix(labs, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["DIFF","SAME"])
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Confusion Matrix @ dist_thr={dist_thr:.3f}")
plt.show()
