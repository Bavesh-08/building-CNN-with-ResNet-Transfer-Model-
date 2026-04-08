<div align="center">

# 🧬 ResNet50V2 + Keras Tuner — CIFAR-10 Classifier

<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
<img src="https://img.shields.io/badge/ResNet50V2-ImageNet-blueviolet?style=for-the-badge" />
<img src="https://img.shields.io/badge/Test Accuracy-92.69%25-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Val Accuracy-93.30%25-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Keras Tuner-Bayesian Optimization-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Colab-Run%20Now-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />

<br/>

> **Fine-tuning ResNet50V2 on CIFAR-10 with automated hyperparameter search via Bayesian Optimization — achieving 92.69% test accuracy.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Hyperparameter Search Space](#-hyperparameter-search-space)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Project Comparison — Full Series](#-project-comparison--full-series)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

This project applies **transfer learning** using **ResNet50V2** (pretrained on ImageNet) to classify CIFAR-10 images, with automated hyperparameter optimization via **Keras Tuner's Bayesian Optimization** engine.

Instead of manually tuning rotation, zoom, dense units, dropout, learning rate, and image resize strategy — we let the tuner search for the optimal combination across 2 trials and pick the best model automatically.

---

## 📦 Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Internal Val Split | 45,000 train / 5,000 val |
| Image Size (native) | 32 × 32 × 3 |
| Upsampled Size | 96 / 128 / 160 (tuned) |
| Classes | 10 |
| Label Format | One-hot encoded |

### 🏷️ Class Labels

```
airplane  •  automobile  •  bird  •  cat  •  deer
dog  •  frog  •  horse  •  ship  •  truck
```

---

## 🏗️ Model Architecture

### Pipeline

```
Input (32×32×3)
    │
    ├── RandomFlip("horizontal")          ← Data Augmentation
    ├── RandomRotation(tuned: 0.05–0.30)
    ├── RandomZoom(tuned: 0.00–0.20)
    │
    ├── Resizing(tuned: 96 / 128 / 160)   ← Upsample for ResNet
    ├── resnet_v2.preprocess_input()       ← Normalize to [-1, 1]
    │
    ├── ResNet50V2 (ImageNet weights)      ← Backbone (trainable)
    │       └── GlobalAveragePooling2D()
    │
    ├── BatchNormalization()
    ├── Dense(tuned: 128–512, relu)        ← Custom Head
    ├── Dropout(tuned: 0.2–0.6)
    │
    └── Dense(10, softmax)                 ← Output
```

> **Note:** The backbone is set `trainable=True` — all ResNet50V2 layers are fine-tuned end-to-end rather than frozen, allowing the model to adapt ImageNet features to CIFAR-10's distribution.

---

## 🎛️ Hyperparameter Search Space

| Hyperparameter | Type | Range / Options |
|---|---|---|
| `rotation` | Float | 0.05 → 0.30 (step 0.05) |
| `zoom` | Float | 0.00 → 0.20 (step 0.05) |
| `img_size` | Choice | 96, 128, 160 |
| `units` | Int | 128 → 512 (step 128) |
| `dropout` | Float | 0.2 → 0.6 (step 0.1) |
| `lr` | Choice | 1e-3, 3e-4, 1e-4 |

**Search Strategy:** Bayesian Optimization — uses a probabilistic model (Gaussian Process) to intelligently select the next trial based on past results, rather than exhaustive grid search.

**Trials Run:** 2 | **Total Search Time:** ~41 minutes

---

## ⚙️ Training Configuration

```python
tuner     = kt.BayesianOptimization(objective="val_accuracy", max_trials=2)
epochs    = 5 (per trial, with early stopping)
batch     = 64
val_split = 0.1  (5,000 images held out internally)
loss      = categorical_crossentropy
optimizer = Adam (lr tuned)
```

**Callbacks:**
```python
EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
```

---

## 📊 Results

### Tuner Search Summary

| Trial | Duration | Val Accuracy |
|:---:|:---:|:---:|
| Trial 1 | ~13 min | 92.xx% |
| **Trial 2** | **~28 min** | **93.30%** ✅ Best |

### Final Evaluation

```
┌────────────────────────────────────────────────┐
│   Best Val Accuracy  :  93.30%                 │
│   Test Accuracy      :  92.69%                 │
│   Test Loss          :  evaluated on 10,000    │
│   Total Tuning Time  :  ~41 minutes            │
│   Trials Completed   :  2                      │
└────────────────────────────────────────────────┘
```

---

## 🏆 Project Comparison — Full Series

> A progression of CIFAR-10 experiments across different architectures and frameworks.

| Metric | 🟨 Custom CNN (Keras) | 🟦 YOLOv8n-cls | 🟥 ResNet50V2 + Tuner (This) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | ~68% | 75.3% | **92.69%** |
| **Val Accuracy** | ~68% | 96.7% | **93.30%** |
| **Parameters** | ~1–2M | 1.45M | ~23.5M |
| **Pretrained Weights** | ❌ | ✅ ImageNet | ✅ ImageNet |
| **Framework** | TensorFlow / Keras | Ultralytics / PyTorch | TensorFlow / Keras |
| **Hyperparameter Tuning** | ❌ Manual | ❌ Manual | ✅ Bayesian Opt |
| **Augmentation** | Basic | RandAugment + HSV | Random + Zoom + Flip |
| **Input Size** | 32×32 | 32×32 | Upsampled to 96–160 |
| **Training Time** | ~15 min | ~6 min | ~41 min (search) |
| **Epochs** | 10–20 | 5 | 5 (per trial) |

### 🔍 Key Takeaways

- 📈 **ResNet50V2 achieves the best accuracy** at 92.69%, a **+17% jump** over YOLOv8 and **+24% over the custom CNN**
- 🧠 **Full fine-tuning** (not freezing the backbone) is the main driver of ResNet's performance gain
- ⚡ **YOLOv8 is the fastest** — 6 min training, 0.66 ms inference, with competitive accuracy for its size
- 🎯 **Bayesian Optimization** removes manual guesswork and finds better configs in fewer trials than grid search
- ⚠️ **Upsampling CIFAR-10** from 32×32 to 96–160px is required for ResNet (designed for 224×224) but adds compute cost
- 🔁 With **more trials and epochs**, ResNet50V2 could push towards 95%+

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install tensorflow keras-tuner
```

### 2. Load & Preprocess Data
```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test,  10)
```

### 3. Define Model Builder + Run Tuner
```python
import keras_tuner as kt

tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=2,
    directory="cifar10_tf_tuning",
    project_name="resnet50v2_pretrained"
)

tuner.search(x_train, y_train, epochs=5, validation_split=0.1,
             callbacks=callbacks, batch_size=64)
```

### 4. Evaluate Best Model
```python
best_model = tuner.get_best_models(1)[0]
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 5. Get Best Hyperparameters
```python
best_hp = tuner.get_best_hyperparameters(1)[0]
print(best_hp.values)
```

---

## 📁 Project Structure

```
resnet50v2-cifar10-tuner/
│
├── cifar10_tf_tuning/
│   └── resnet50v2_pretrained/
│       ├── trial_0/                  # Trial 1 artifacts & weights
│       └── trial_1/                  # Trial 2 artifacts & weights (best)
│
├── data/                             # CIFAR-10 auto-downloaded by Keras
│
└── notebook.ipynb                    # Full training + tuning notebook (Colab)
```

---

## 🧪 Future Improvements

- [ ] Increase `max_trials` to 10–15 for a deeper search
- [ ] Try `EfficientNetV2` or `ConvNeXt` backbone for better accuracy/efficiency
- [ ] Add `CosineDecayRestarts` learning rate schedule
- [ ] Freeze backbone first → train head → unfreeze for fine-tuning (two-phase approach)
- [ ] Use `Hyperband` tuner for faster convergence on larger search spaces
- [ ] Export best model to TFLite / ONNX for edge deployment

---

## 👨‍💻 Author

**Bavesh V** — ML & AI Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

---

<div align="center">
  <sub>Part of an ongoing series of ML benchmark experiments — MNIST → CIFAR-10 (Custom CNN) → CIFAR-10 (YOLOv8) → CIFAR-10 (ResNet50V2 + Tuner)</sub>
</div>
