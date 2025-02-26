# PSAD: Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection

This repository includes a pytorch implementation of the paper "Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection" accepted in [AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/28703).

## Abstract

Logical anomalies (LA) refer to data violating underlying logical constraints e.g., the quantity, arrangement, or composition of components within an image. Detecting accurately such anomalies requires models to reason about various component types through segmentation. However, curation of pixel-level annotations for semantic segmentation is both time-consuming and expensive. Although there are some prior few-shot or unsupervised co-part segmentation algorithms, they often fail on images with industrial object. These images have components with similar textures and shapes, and a precise differentiation proves challenging. In this study, we introduce a novel component segmentation model for LA detection that leverages a few labeled samples and unlabeled images sharing logical constraints. To ensure consistent segmentation across unlabeled images, we employ a histogram matching loss in conjunction with an entropy loss. As segmentation predictions play a crucial role, we propose to enhance both local and global sample validity detection by capturing key aspects from visual semantics via three memory banks: class histograms, component composition embeddings and patch-level representations. For effective LA detection, we propose an adaptive scaling strategy to standardize anomaly scores from different memory banks in inference. Extensive experiments on the public benchmark MVTec LOCO AD reveal our method achieves 98.1% AUROC in LA detection vs. 89.6% from competing methods.

## Few Shot Part Segmentation
![fig_fss](https://github.com/oopil/PSAD_logical_anomaly_detection/assets/44998223/68056d95-f62e-438c-804b-e3c9b001e018)

## PSAD
![fig_psad](https://github.com/oopil/PSAD_logical_anomaly_detection/assets/44998223/8e0193f3-d713-4c11-b43a-c14163ffb99f)

## Dataset
- We use MVTec LOCO AD Dataset for evaluation. For details of this dataset, please see [here](https://www.mvtec.com/company/research/datasets/mvtec-loco). 

## Training 
To implement our proposed Part Segmentation-based Anomaly Detection (PSAD), follow procedure shown as below. If you want to skip the preprocessing and segmentation process 1~4, download [preprocessed images](https://drive.google.com/file/d/1lpJgU2G5cpW6b_WTtKJUNH7Oelyltioc/view?usp=drive_link), [segmentation maps](https://drive.google.com/file/d/1Esa06exQ2cH3c3GIozpdU2-qmwKiVBdr/view?usp=sharing) and use them as inputs.

1. Annotate few labeled images per category.

We used an [annotation tool](https://www.makesense.ai/) to make annotations for few images. In most cases, we used 5 labeled images. But when there are multiples product types (e.g., juicie bottle and splicing connectors), we used 1 labeled images per type, i.e., total 3 images.

2. Preprocess images. 

Convert label format, crop and resize the image. Please check the code in `preprocess` directory.

3. Train a few shot part segmentation model using few labeld and numerous unlabeled images. Using different levels of features of pretrained encoder can lead to different segmentation results.
```
CUDA_VISIBLE_DEVICES=[GPU_ID] python finetune_cnn_coord.py 
    --n_shot 5 
    --num_epochs 100 
    --obj_name [OBJ_NAME] 
    --snapshot_dir [DIR]
```

4. Train a segmentation model again using predicted pseudo label.
```
run_unet.sh [GPU_ID] [DIR]
```

5. Save predictions of [PatchCore](https://github.com/amazon-science/patchcore-inspection). Please check `patchcore` directory. Or you can use the [scores](https://drive.google.com/file/d/1Q8RVR8rDV6oOMhRa_8fEYBM9OVQIn2eM/view?usp=drive_link) already obtained using PatchCore. It includes anomaly scores of all train and test data predicted using PatchCore without the coreset sampling, and the scores of train data by treating each as a test data for 'adaptive scaling'. Each .pth file (Ex. dict = train/good/000.pt) has a dictionary with 5 keys ['distance', 'anomaly_map_interpolate', 'anomaly_maps', 'anomaly_scores', 'label', 'name']. 'anomaly_scores' is the score for the adaptive scaling.

6. Train and test PSAD based on segmentation results. You can choose memory type like "hcp". h,c, and p denote histogram, composition, and patch memory banks.
```
./run_ad.sh [GPU_ID] [DIR] "hcp" "max"
```

## Testing
Our proposed PSAD and other comparison methods are evaluated using image AUROC scores. As there is no ground truth for segmentation task, we indirectly evaluated the segmentation performance using anomaly detection performance. In general, accurate segmentation correlates with anomaly detection performance.

## Results
- Comparison with state-of-the-art methods

| Category |                       | PatchCore | RD4AD | DRAEM |   ST  |  AST  |  GCAD  | SINBAD | ComAD |  SLSG |  PSAD  |
|:--------:|:---------------------:|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:------:|:-----:|:-----:|:------:|
|    LA    | Breakfast Box         |   74.8    | 66.7  | 75.1  | 68.9  | 80.0  |  87.0  |  96.5  | 91.1  |   -   | 100.0  |
|          | Juice Bottle          |   93.9    | 93.6  | 97.8  | 82.9  | 91.6  | 100.0  |  96.6  | 95.0  |   -   |  99.1  |
|          | Pushpins              |   63.6    | 63.6  | 55.7  | 59.5  | 65.1  |  97.5  |  83.4  | 95.7  |   -   | 100.0  |
|          | Screw Bag             |   57.8    | 54.1  | 56.2  | 55.5  | 80.1  |  56.0  |  78.6  | 71.9  |   -   |  99.3  |
|          | Splicing   Connectors |   79.2    | 75.3  | 75.2  | 65.4  | 81.8  |  89.7  |  89.3  | 93.3  |   -   |  91.9  |
|          | Average (LA)          |   74.0    | 70.7  | 72.0  | 66.4  | 79.7  |  86.0  |  88.9  | 89.4  | 89.6  |  98.1  |
|    SA    | Breakfast Box         |   80.1    | 60.3  | 85.4  | 68.4  | 79.9  |  80.9  |  87.5  | 81.6  |   -   |  84.9  |
|          | Juice Bottle          |   98.5    | 95.2  | 90.8  | 99.3  | 95.5  |  98.9  |  93.1  | 98.2  |   -   |  98.2  |
|          | Pushpins              |   87.9    | 84.8  | 81.5  | 90.3  | 77.8  |  74.9  |  74.2  | 91.1  |   -   |  89.8  |
|          | Screw Bag             |   92.0    | 89.2  | 85.0  | 87.0  | 95.9  |  70.5  |  92.2  | 88.5  |   -   |  95.7  |
|          | Splicing   Connectors |   88.0    | 95.9  | 95.5  | 96.8  | 89.4  |  78.3  |  76.7  | 94.9  |   -   |  89.3  |
|          | Average (SA)          |   89.3    | 85.1  | 87.6  | 88.4  | 87.7  |  80.7  |  84.7  | 90.9  | 91.4  |  91.6  |
|  Average |                       |   81.7    | 77.9  | 79.8  | 77.4  | 83.7  |  83.4  |  86.8  | 90.1  | 90.3  |  94.0  |


- PSAD performance depending on segmentation model

| Models                            | LA   | SA   |
|-----------------------------------|------|------|
| SCOPS (Hung et al. 2019)          | 82.5 | 90.2 |
| Part-Assembly (Gao et al.   2021) | 80.3 | 85.6 |
| SegGPT (Wang et al. 2023)         | 88.7 | 87.2 |
| VAT (Hong et al. 2022)            | 79.2 | 87.8 |
| RePRI (Boudiaf et al. 2021)       | 83.6 | 88.4 |
| Ours (L_sup)                      | 95.9 | 89.6 |
| Ours (L_sup + L_H))               | 96.3 | 90   |
| Ours (L_sup + L_H + L_hist)       | 98.1 | 91.6 |

- Qualitative evaluation on FSS models
![fss_seg](https://github.com/oopil/PSAD_logical_anomaly_detection/assets/44998223/6cb07231-d4d3-4dff-a576-13743008ab38)

- Ablation study on multiple memory banks and adaptive scaling (AS)

| M_hist | M_comp | M_patch | AS |  LA  |  SA  |
|:------:|:------:|:-------:|:--:|:----:|:----:|
|    ✓   |        |         |    | 94.2 | 71.1 |
|        |    ✓   |         |    | 90.9 | 85.4 |
|        |        |    ✓    |    | 73.9 | 89.3 |
|    ✓   |    ✓   |    ✓    |    | 96.8 | 87.6 |
|    ✓   |    ✓   |    ✓    |  ✓ | 98.1 | 91.6 |

- PSAD performance using less normal images (same segmentation model is used.)

|    N_M    | 100% |  50% |  25% | 12.5% |
|:---------:|:----:|:----:|:----:|:-----:|
| Avg AUROC | 97.4 | 97.1 | 96.6 |  96.2 |

- Visualizing hitrograms from different memory banks
![qual_hist](https://github.com/oopil/PSAD_logical_anomaly_detection/assets/44998223/d299e1ac-6683-42f6-b446-9835adbe01d2)

<!-- ## Citing
```

``` -->
## Acknowledgments
Our work was inspired by many previous works related to industrial anomaly detection and few shot segmentation including [PatchCore](https://github.com/amazon-science/patchcore-inspection), [RePRI](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation/tree/master). Thanks to their inspiring works.
