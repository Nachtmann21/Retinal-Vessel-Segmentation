import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    confusion_matrix as cm_fn,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# ——— 0. Configuration ————————————————————————————————————————————————
root_dir = 'Drive Segmented'
IMG_SIZE  = 224
N_PAIRS   = 1000   # number of random pairs to sample
TPR_TARGETS = [0.90, 0.95, 0.99]  # thresholds at these TPRs

# custom loss for loading
def contrastive_loss(y_true, y_pred, margin=1.0):
    sq = tf.square(y_pred)
    md = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * sq + (1 - y_true) * md)

# ——— 1. Load models —————————————————————————————————————————————————————
siamese_model = tf.keras.models.load_model(
    'best_siamese.h5',
    custom_objects={'contrastive_loss': contrastive_loss}
)
embed_net = tf.keras.models.load_model('embed_net.h5')

# ——— 2. Helpers ————————————————————————————————————————————————————————
persons = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]

def sample_pair(same: bool):
    if same:
        pid = random.choice(persons)
        imgs = os.listdir(os.path.join(root_dir, pid))
        a, b = random.sample(imgs, 2)
        lbl = 1
        pA = os.path.join(root_dir, pid, a)
        pB = os.path.join(root_dir, pid, b)
    else:
        p1, p2 = random.sample(persons, 2)
        a = random.choice(os.listdir(os.path.join(root_dir, p1)))
        b = random.choice(os.listdir(os.path.join(root_dir, p2)))
        lbl = 0
        pA = os.path.join(root_dir, p1, a)
        pB = os.path.join(root_dir, p2, b)
    return (
        Image.open(pA).convert('L'),
        Image.open(pB).convert('L'),
        lbl
    )

def get_test_pairs(n=N_PAIRS):
    dists, labs = [], []
    for _ in range(n):
        imgA, imgB, lbl = sample_pair(random.random() < 0.5)
        arrA = np.expand_dims(np.array(imgA, np.float32)/255., -1)
        arrB = np.expand_dims(np.array(imgB, np.float32)/255., -1)
        tA = tf.image.resize(tf.convert_to_tensor(arrA), [IMG_SIZE,IMG_SIZE])
        tB = tf.image.resize(tf.convert_to_tensor(arrB), [IMG_SIZE,IMG_SIZE])
        batchA = tf.expand_dims(tA,0)
        batchB = tf.expand_dims(tB,0)
        dist = siamese_model.predict([batchA, batchB], verbose=0)[0,0]
        dists.append(dist)
        labs.append(lbl)
    return np.array(dists), np.array(labs)

# ——— 3. ROC & AUC ———————————————————————————————————————————————————————
dists, labs = get_test_pairs()
fpr, tpr, thr = roc_curve(labs, -dists)
roc_auc = auc(fpr, tpr)
print(f"\n>> AUC: {roc_auc:.4f}\n")

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); plt.show()

# ——— 4. Thresholds & Metrics ———————————————————————————————————————————
results = []
# a) thresholds at fixed TPR targets
for target in TPR_TARGETS:
    idx = np.argmax(tpr >= target)
    dist_thr = -thr[idx]
    y_pred = (dists < dist_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labs, y_pred, labels=[0,1]).ravel()
    prec = precision_score(labs, y_pred)
    rec  = recall_score(labs, y_pred)
    f1   = f1_score(labs, y_pred)
    results.append({
        'Method':   f"TPR≥{int(target*100)}%",
        'Thresh':   dist_thr,
        'TPR':      tp/(tp+fn),
        'FPR':      fp/(fp+tn),
        'Precision':prec,
        'Recall':   rec,
        'F1':       f1,
        'TN':       tn, 'FP': fp, 'FN': fn, 'TP': tp
    })

# b) Youden’s J (max TPR−FPR)
youden = tpr - fpr
idxJ = np.argmax(youden)
distJ = -thr[idxJ]
y_pred = (dists < distJ).astype(int)
tn, fp, fn, tp = confusion_matrix(labs, y_pred, labels=[0,1]).ravel()
results.append({
    'Method':   "Youden’s J",
    'Thresh':   distJ,
    'TPR':      tpr[idxJ],
    'FPR':      fpr[idxJ],
    'Precision':precision_score(labs, y_pred),
    'Recall':   recall_score(labs, y_pred),
    'F1':       f1_score(labs, y_pred),
    'TN':       tn, 'FP': fp, 'FN': fn, 'TP': tp
})

# Print a neat table
print("=== Threshold & Performance Summary ===")
print(f"{'Method':<12} {'Thresh':>7}  {'TPR':>5}  {'FPR':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}")
for r in results:
    print(f"{r['Method']:<12} {r['Thresh']:7.3f}  "
          f"{r['TPR']*100:5.1f}%  {r['FPR']*100:5.1f}%  "
          f"{r['Precision']*100:5.1f}%  {r['Recall']*100:5.1f}%  {r['F1']*100:5.1f}%")
print()

# ——— 5. (Optional) Display one confusion matrix ——————————————————————
best = results[0]  # e.g. TPR≥90%; pick whichever you prefer
y_pred = (dists < best['Thresh']).astype(int)
cm = confusion_matrix(labs, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["DIFF","SAME"])
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Confusion Matrix @ {best['Method']} (thr={best['Thresh']:.3f})")
plt.show()
