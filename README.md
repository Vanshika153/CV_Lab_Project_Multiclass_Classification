# CV_Lab_Project_Multiclass_Classification
# Skin Lesion Classification using EfficientNet-B2 and ResNet-50

## About the Project

Skin lesion classification is an important application of deep learning in medical image analysis. However, building a model that can accurately identify different types of skin lesions is challenging, especially when the dataset contains an unequal number of images for different classes.

In this project, we developed a deep learning-based skin lesion classification system that uses **EfficientNet-B2 and ResNet-50 together** to classify dermoscopic images into seven different categories.

Instead of relying on a single model, we combined the predictions of both networks using an ensemble approach. We also focused on handling class imbalance, avoiding data leakage, and testing how well the model performs on images from a different dataset.

Our main goal was to build a model that not only performs well on the training dataset but can also generalize to previously unseen images.

## What We Tried to Achieve

- Classify skin lesions into seven different categories.
- Improve classification performance by combining two CNN architectures.
- Handle class imbalance using weighted sampling and data augmentation.
- Reduce data leakage through lesion-based dataset splitting.
- Test the model on an external dataset to check its generalization ability.
- Compare our approach with an implementation of EfficientCapsNet.
- Understand model predictions using Grad-CAM visualizations.

## Models Used

We used two pretrained deep learning models and combined their predictions.

**1. EfficientNet-B2**

EfficientNet-B2 is used to extract important visual features from skin lesion images. We used ImageNet-pretrained weights and replaced its final classification layer to predict the seven lesion classes.

**2. ResNet-50**

ResNet-50 is another pretrained CNN architecture that uses residual connections to learn complex image features. Its final classification layer was also modified for our seven-class problem.

**3. Ensemble Model**

Both models were trained independently. Their prediction probabilities were then averaged to obtain the final prediction.

The formula used was:

`Final Prediction = (EfficientNet-B2 + ResNet-50) / 2`

This allowed us to combine the predictions of two different architectures without introducing additional tuned ensemble weights.

## Dataset

We worked with two publicly available dermoscopic image datasets:

**HAM10000**
- Contains 10,015 dermoscopic images.
- Includes seven different skin lesion categories.
- Used for training, validation, and initial testing.

**ISIC 2019**
- Originally contains 25,331 images across eight categories.
- The SCC category was excluded to maintain consistency with the seven classes in HAM10000.
- Duplicate images already present in HAM10000 were removed.
- The remaining 14,885 images were used for external testing.

### Classes Covered

| Label | Category |
|---|---|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis-like Lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

## How We Approached the Problem

**1. Image Preprocessing**

Before feeding images into the models, we:
- Resized images to 224 × 224 pixels.
- Applied Shades-of-Gray colour constancy to reduce lighting and colour variations.
- Normalized pixel values.
- Applied ImageNet normalization for both models.

**2. Dataset Splitting**

We divided HAM10000 into training, validation, and testing sets using lesion-based grouping and class stratification.

| Split | Number of Images |
|---|---:|
| Training | 7,153 |
| Validation | 1,431 |
| Testing | 1,431 |

This helped ensure that images belonging to the same lesion were not distributed across different splits.

**3. Data Augmentation**

Since medical image datasets often contain limited samples for certain classes, we applied augmentation to the training images.

Some techniques included:
- Horizontal and vertical flipping.
- Rotation and scaling.
- Random affine transformations.
- Brightness, contrast, and colour adjustments.

**4. Handling Class Imbalance**

Not all skin lesion categories had the same number of images. To avoid the model becoming biased towards frequently occurring classes, we used a weighted random sampler based on class frequency.

**5. Model Training**

Both models were trained independently using the following setup:

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Loss Function | Cross-Entropy with Label Smoothing |
| Scheduler | Cosine Annealing |
| Maximum Epochs | 50 |
| Early Stopping Patience | 10 |

The best model checkpoints were selected using validation balanced accuracy.

**6. Model Explainability**

We also used Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image influenced the model's predictions.

This helped us examine whether the models were focusing on the lesion regions while making predictions.

## Results

We compared our proposed ensemble with a re-implementation of EfficientCapsNet.

The following results were obtained:

| Dataset | EfficientCapsNet | Our Ensemble |
|---|---:|---:|
| HAM10000 Test | 40.53% | **83.10%** |
| ISIC 2019 External | 28.87% | **60.74%** |
| Combined Test | 29.89% | **62.70%** |

### Other Evaluation Metrics

| Dataset | Balanced Accuracy | Macro F1-Score |
|---|---:|---:|
| HAM10000 | 63.55% | 66.28% |
| ISIC 2019 | 43.18% | 42.70% |
| Combined | 44.94% | 44.79% |

### What We Observed

- The ensemble performed better than the re-implemented EfficientCapsNet on all three evaluation datasets.
- Combining EfficientNet-B2 and ResNet-50 improved accuracy compared to using either network individually.
- Performance dropped on the external ISIC 2019 dataset, showing that domain differences between datasets remain a major challenge.
- Grad-CAM helped us visualize the regions that contributed to the model's predictions.

*Note: The comparison is with our re-implementation of EfficientCapsNet under the same evaluation protocol. It should not be interpreted as a direct comparison with the 72.75% accuracy reported in the original paper.*

## Technologies Used

- Python
- PyTorch
- Torchvision
- TIMM (PyTorch Image Models)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM

## Limitations and Future Improvements

Although the ensemble showed promising results, there are still areas that can be improved.

Some limitations of the current work include:
- The experiments were conducted using a single random seed and dataset split.
- Performance decreased when evaluated on the external dataset.
- Some lesion categories had very few test images.
- Further testing is needed to establish how well the model generalizes across different clinical settings.
- The model has not undergone clinical validation.

In the future, we would like to explore:
- Combining CNN architectures with Vision Transformers.
- Improving generalization across different datasets.
- Experimenting with stronger data augmentation techniques.
- Testing the model on additional external datasets.
- Performing more extensive cross-validation and ablation studies.
- Studying model efficiency and inference time.

## Authors

**Priyanshi Goel**  
**Vanshika Goyal**

Computer Science and Engineering (Artificial Intelligence)  
Netaji Subhas University of Technology (NSUT), New Delhi.

