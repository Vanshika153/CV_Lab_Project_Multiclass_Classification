
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

### Model Architecture Visualization

The following diagram shows the overall architecture of our proposed ensemble model, combining EfficientNet-B2 and ResNet-50.

<img width="877" height="580" alt="image" src="https://github.com/user-attachments/assets/5dbb9994-2084-4c97-8655-b3b52974d9cc" />


## Dataset

We worked with two publicly available dermoscopic image datasets:

**[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)**
- Contains 10,015 dermoscopic images.
- Includes seven different skin lesion categories.
- Used for training, validation, and initial testing.

**[ISIC 2019](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification?resource=download)**
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
- Applied Shades-of-Gray colour constancy to reduce lighting and colour variations [11].
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
| Optimizer | AdamW [12] |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Loss Function | Cross-Entropy with Label Smoothing [13] |
| Scheduler | Cosine Annealing |
| Maximum Epochs | 50 |
| Early Stopping Patience | 10 |

The best model checkpoints were selected using validation balanced accuracy.

**6. Model Explainability**

We also used Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image influenced the model's predictions [15].

This helped us examine whether the models were focusing on the lesion regions while making predictions.

### Grad-CAM Visualizations

The visualizations below show the regions of the skin lesion images that influenced the model's predictions across the seven lesion categories.

BASE MODEL
<img width="1280" height="377" alt="image" src="https://github.com/user-attachments/assets/d364edc8-a596-4730-b761-2863c783e02a" />

EffNetB2 + ResNet50
<img width="1280" height="387" alt="image" src="https://github.com/user-attachments/assets/4c05f28e-8c6c-4bdf-aa32-3c403e7f493b" />

## Results

We compared our proposed ensemble with a re-implementation of EfficientCapsNet [1].

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

### Confusion Matrices

The confusion matrices provide a class-wise view of the ensemble's predictions on the HAM10000 test set, external ISIC 2019 dataset, and combined evaluation set.

- BASE
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/2793d964-fd01-400b-92c5-fa429adaab1e" />
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/02cdbe2a-c758-4e29-bc61-93a7f19c6139" />
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/11153793-40f1-47d3-9e6d-d114347bac29" />


- EffNetB2 + ResNet50
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/067bfb52-bb0a-40c6-b82c-26b606286178" />
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/618de51c-0228-4ee4-8a2f-46fab46950ca" />
<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/a5140e4d-4773-4a15-9a26-5cb80b983d7f" />

### Training and Validation Curves

The following graphs show the training and validation behaviour of the models, including training loss, validation accuracy, and validation balanced accuracy.

- BASE MODEL
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/40cc5f20-5346-4769-9f83-524e2fa5c55e" />
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/a8126749-bd97-4613-9197-f2b413a0c31e" />


- EffNetB2 + ResNet50
<img width="1280" height="389" alt="image" src="https://github.com/user-attachments/assets/50117fac-c3c1-4dbb-8936-708160fccb60" />


### What We Observed

- The ensemble performed better than the re-implemented EfficientCapsNet on all three evaluation datasets.
- Combining EfficientNet-B2 and ResNet-50 improved accuracy compared to using either network individually.
- Performance dropped on the external ISIC 2019 dataset, showing that domain differences between datasets remain a major challenge.
- Grad-CAM helped us visualize the regions that contributed to the model's predictions.

*Note: The comparison is with our re-implementation of EfficientCapsNet under the same evaluation protocol. It should not be interpreted as a direct comparison with the 72.75% accuracy reported in the original paper [1].*

## Repository contents

- `Results_Base_Model/` – Evaluation results of the base model.
- `Results_Proposed_Model/` – Evaluation results of the proposed model.

### Trained on HAM10000 + ISIC 2019 (combined)
Each notebook is tested separately on the held-out, lesion-disjoint test splits of HAM10000 and ISIC 2019.
- `EfficientCapsNet_combined.ipynb` – EfficientCapsNet.
- `EffNetB2_ResNet50_combined.ipynb` – EfficientNet-B2 + ResNet-50 ensemble with data-leakage control (**our proposed model**).

### Trained on HAM10000 only (external test)
Each notebook is tested on ISIC 2019 and on the combined HAM10000 + ISIC 2019 data.
- `EfficientCapsNet_test.ipynb` – EfficientCapsNet.
- `EffNetB2_ResNet50_test.ipynb` – EfficientNet-B2 + ResNet-50 ensemble with data-leakage control (**our proposed model**).
- `EffNetB2_ResNet50_Calibrated_test.ipynb` – Calibrated EfficientNet-B2 + ResNet-50 ensemble (Keras/TensorFlow).
- `EffNetB2_DenseNet121_test.ipynb` – EfficientNet-B2 + DenseNet121 ensemble with data-leakage control.
  
## Data Folder Layout
Place the data in the project folder before running any notebook:

```data/
├── HAM10000_metadata.csv
├── images/                      # HAM10000 .jpg files
└── ISIC2019_Dataset/
    ├── AK/ BCC/ BKL/ DF/ MEL/ NV/ VASC/
    └── ISIC_2019_Training_Metadata.csv
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- TIMM (PyTorch Image Models) [14]
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM

## Project Structure

```CV_Lab_Project_Multiclass_Classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── EffNetB2_ResNet50_test.ipynb              # proposed model: train + HAM10000 / ISIC2019 evaluation + Grad-CAM
├── EffNetB2_ResNet50_combined.ipynb          # proposed model trained on combined HAM10000 + ISIC2019
├── EffNetB2_ResNet50_Calibrated_test.ipynb   # ensemble with probability calibration
├── EffNetB2_DenseNet121_test.ipynb           # alternative ensemble (EfficientNet-B2 + DenseNet121)
├── EfficientCapsNet_test.ipynb               # base model: EfficientCapsNet on HAM10000
├── EfficientCapsNet_combined.ipynb           # base model: EfficientCapsNet on combined data
│
├── Results_Base_Model/
│   ├── accuracy_loss_curves.png
│   ├── base_cm_combined.png
│   ├── base_cm_ham10000_test.png
│   ├── base_cm_isic2019_test.png
│   ├── base_gradcam.png
│   ├── base_roc_ham10000_test.png
│   └── base_roc_isic2019_test.png
│
├── Results_Proposed_Model/
│   ├── cm_ens_combined.png
│   ├── cm_ens_ham10000.png
│   ├── cm_ens_isic2019.png
│   ├── ens_curves.png
│   └── gradcam_sample.png
│
├── data/                    # not in repo: download separately
│   ├── HAM10000_metadata.csv
│   ├── images/              # HAM10000 images
│   └── ISIC2019_Dataset/
│
├── models/                  # not in repo: created during training (or download weights)
│   └── *.pth
│
└── research_results/        # not in repo: created by the notebooks (split CSVs, plots)
```

## Limitations and Future Improvements

Although the ensemble showed promising results, there are still areas that can be improved.

Some limitations of the current work include:
- The experiments were conducted using a single random seed and dataset split.
- Performance decreased when evaluated on the external dataset.
- Some lesion categories had very few test images.
- Further testing is needed to establish how well the model generalizes across different clinical settings.
- The model has not undergone clinical validation.

In the future, we would like to explore:
- Combining CNN architectures with Vision Transformers, such as Swin Transformer [4].
- Improving generalization across different datasets.
- Experimenting with stronger data augmentation techniques.
- Testing the model on additional external datasets.
- Performing more extensive cross-validation and ablation studies.
- Studying model efficiency and inference time.

## References

1. Afriyie, Y., Katherina, A. F. A., & Ussiph, N. (2026). *[Robust skin lesion classification through enhanced minority class recognition using hybrid EfficientNet–Capsule networks.](https://www.sciencedirect.com/science/article/pii/S2666521226001456)* Intelligence-Based Medicine, 16, 100487.

2. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* Proceedings of ICML.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* Proceedings of CVPR.

4. Liu, Z., et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* Proceedings of ICCV.

5. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks.* Proceedings of CVPR.

6. Sabour, S., Frosst, N., & Hinton, G. (2017). *Dynamic routing between capsules.* Advances in Neural Information Processing Systems (NeurIPS).

7. Lin, T. Y., et al. (2017). *Focal loss for dense object detection.* Proceedings of ICCV.

8. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *[The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

9. Codella, N. C. F., et al. (2017). *Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC).* arXiv:1710.05006.

10. Combalia, M., et al. (2019). *BCN20000: Dermoscopic lesions in the wild.* arXiv:1908.02288.

11. Finlayson, G. D., & Trezzi, E. (2004). *Shades of gray and colour constancy.* Proceedings of the Color and Imaging Conference.

12. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* Proceedings of ICLR.

13. Szegedy, C., et al. (2016). *Rethinking the inception architecture for computer vision.* Proceedings of CVPR.

14. Wightman, R. (2019). *PyTorch Image Models (timm).* GitHub repository.

15. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* Proceedings of ICCV, pp. 618–626.

16. Esteva, A., et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks.* Nature, 542, 115–118.

## Authors

**Priyanshi Goel**  
**Vanshika Goyal**

Computer Science and Engineering (Artificial Intelligence)  
Netaji Subhas University of Technology (NSUT), New Delhi.

## Disclaimer

This project was developed for academic and research purposes. It is an experimental deep learning model and is not intended to replace professional medical advice, diagnosis, or treatment.
