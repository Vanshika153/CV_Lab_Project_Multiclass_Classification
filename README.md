# CV_Lab_Project_Multiclass_Classification
# Skin Lesion Classification Using EfficientNet-B2 and ResNet-50 Ensemble

## 📌 Project Overview

This project presents a deep learning-based approach for **seven-class skin lesion classification** using a heterogeneous ensemble of EfficientNet-B2 and ResNet-50.

The primary objective is to improve skin lesion classification performance while addressing challenges such as class imbalance, data leakage, and domain shift between different dermoscopic image datasets.

The proposed model combines predictions from two independently trained convolutional neural networks using an equal-weight probability averaging strategy.

The ensemble is evaluated against a re-implementation of EfficientCapsNet under a leakage-controlled evaluation protocol, including external testing on unseen ISIC 2019 images.

## 🎯 Objectives

- Develop an effective deep learning model for multiclass skin lesion classification.
- Improve classification performance using an EfficientNet-B2 and ResNet-50 ensemble.
- Address class imbalance through training-only weighted sampling and augmentation.
- Prevent data leakage using lesion-grouped, class-stratified dataset splitting.
- Evaluate generalization using an independent ISIC 2019 external test set.
- Compare the proposed ensemble against EfficientCapsNet and other ensemble configurations.
- Use Grad-CAM to visualize model attention and improve interpretability.

## 🧠 Model Architecture

The proposed architecture consists of two independently trained pretrained CNN models:

### 1. EfficientNet-B2
- Uses ImageNet-pretrained weights.
- Classifier replaced with a seven-class linear classification layer.
- Extracts discriminative visual features from skin lesion images.
- Contains approximately 7.71 million parameters.

### 2. ResNet-50
- Uses ImageNet-pretrained weights.
- Final fully connected layer replaced with a seven-class classifier.
- Uses residual connections to learn deeper image representations.
- Contains approximately 23.52 million parameters.

### 3. Ensemble Strategy

The softmax probability outputs of both models are combined using an equal-weight average:

`P_final = (P_EfficientNet-B2 + P_ResNet-50) / 2`

The final predicted class is selected using the argmax operation.

No additional tuned ensemble weights, thresholds, or class multipliers are used in the final model.

## 📂 Datasets

The project uses two publicly available dermoscopic image datasets:

### HAM10000
- Total images: 10,015
- Seven skin lesion classes.
- Used for model training, validation, and in-dataset testing.

### ISIC 2019
- Original images: 25,331
- Squamous Cell Carcinoma (SCC) class excluded to maintain a seven-class classification task.
- Images overlapping with HAM10000 removed using image IDs.
- Remaining 14,885 images used exclusively for external evaluation.

### Classification Categories

| Label | Skin Lesion Category |
|---|---|
| akiec | Actinic Keratoses and Intraepithelial Carcinoma |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis-like Lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

## ⚙️ Methodology

### 1. Data Preprocessing
- Resize images to 224 × 224 pixels.
- Apply Shades-of-Gray colour constancy (Minkowski p = 6).
- Normalize pixel values to the range [0, 1].
- Apply ImageNet normalization separately for each backbone.

### 2. Data Splitting

HAM10000 is divided using a lesion-grouped, class-stratified split with random seed 42.

| Dataset Split | Images |
|---|---:|
| Training | 7,153 |
| Validation | 1,431 |
| Testing | 1,431 |

No lesion is shared between the three partitions.

### 3. Data Augmentation

Augmentation is applied exclusively to the training dataset.

Techniques include:
- Horizontal and vertical flipping.
- Random affine transformations.
- Rotation (±30°).
- Translation (10%).
- Scaling (0.85–1.15).
- Shearing (10°).
- Colour jitter for brightness, contrast, saturation, and hue.

### 4. Class Imbalance Handling

A weighted random sampler is used to address the imbalance between lesion categories.

Sampling weights are inversely proportional to class frequency and calculated using the training split only.

### 5. Model Training

Both networks are independently trained using:
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Batch size: 32
- Loss function: Cross-Entropy with label smoothing (0.1)
- Learning rate scheduler: Cosine Annealing
- Maximum epochs: 50
- Early stopping: Patience of 10 epochs
- Random seed: 42

The best checkpoint is selected using validation balanced accuracy.

### 6. Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the image regions influencing model predictions.

Grad-CAM visualizations are generated for both EfficientNet-B2 and ResNet-50 to examine their attention towards lesion regions.

## 📊 Experimental Results

The proposed ensemble is compared with the re-implemented EfficientCapsNet under the same HAM10000-only training protocol.

### Performance Comparison

| Evaluation Dataset | EfficientCapsNet Accuracy | Proposed Ensemble Accuracy |
|---|---:|---:|
| HAM10000 Test | 40.53% | **83.10%** |
| ISIC 2019 External | 28.87% | **60.74%** |
| Combined Test | 29.89% | **62.70%** |

### Additional Metrics

| Dataset | Balanced Accuracy | Macro F1-Score |
|---|---:|---:|
| HAM10000 | 63.55% | 66.28% |
| ISIC 2019 | 43.18% | 42.70% |
| Combined | 44.94% | 44.79% |

The proposed ensemble achieves improvements over the re-implemented EfficientCapsNet across all three evaluation datasets.

**Key findings:**
- Achieved 83.10% accuracy on the HAM10000 held-out test set.
- Achieved 60.74% accuracy on 14,885 unseen ISIC 2019 images.
- Improved external test accuracy by approximately 31.87 percentage points over the re-implemented baseline.
- Outperformed the compared models in accuracy across all three evaluation sets.
- Demonstrated that external domain shift remains a significant challenge for skin lesion classification.

*Note: These results are specific to the experimental protocol described in the research paper. The reported comparison is against our re-implementation of EfficientCapsNet, not a direct reproduction of the original paper's reported 72.75% accuracy.*

## 🔍 Evaluation Metrics

The following metrics are used to evaluate model performance:

- **Accuracy:** Measures the proportion of correctly classified images.
- **Balanced Accuracy:** Accounts for class imbalance by averaging recall across classes.
- **Precision:** Measures the proportion of correct positive predictions.
- **Recall:** Measures how many actual class instances are correctly identified.
- **Macro F1-Score:** Calculates the average F1-score across all classes.
- **Weighted F1-Score:** Calculates the F1-score weighted by class support.

Additional evaluation includes:
- Confusion matrices.
- One-vs-rest ROC curves.
- Training and validation curves.
- Double-fault and disagreement rates.
- Grad-CAM visualizations.

## ⚠️ Limitations

- Experiments use a single random seed and dataset split.
- Performance decreases on the external ISIC 2019 dataset due to domain shift.
- Minority classes have relatively few test samples.
- Same-patient leakage across different lesions cannot be completely excluded because lesion IDs are not patient IDs.
- No clinical validation has been performed.
- Model inference latency and computational complexity have not been benchmarked.
- The task is limited to seven lesion categories.

## 🚀 Future Scope

Potential improvements include:

- Exploring hybrid CNN-Transformer architectures such as EfficientNet-B2 + Swin Transformer.
- Improving domain generalization through stronger augmentation and domain-robust training.
- Evaluating additional external datasets.
- Applying perceptual-hash-based near-duplicate filtering.
- Performing repeated grouped cross-validation with confidence intervals.
- Conducting controlled ablation studies.
- Measuring inference latency and computational efficiency.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- PyTorch Image Models (timm)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM

## 👩‍💻 Authors

**Priyanshi Goel**  
**Vanshika Goyal**

Computer Science and Engineering (Artificial Intelligence)  
Netaji Subhas University of Technology (NSUT), New Delhi, India.

## 📚 References

1. Afriyie et al., *Robust Skin Lesion Classification Through Enhanced Minority Class Recognition Using Hybrid EfficientNet–Capsule Networks*, Intelligence-Based Medicine, 2026.
2. Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML, 2019.
3. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
4. Tschandl et al., *The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions*, Scientific Data, 2018.
5. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV, 2017.

---

**Disclaimer:** This project is developed for academic and research purposes only. It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical evaluation.
