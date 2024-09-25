# Breast cancer detection using Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

## **Introduction and Context**

This dataset comes from the analysis of breast tumors using Fine Needle Aspiration (FNA) biopsy, where tissue samples are observed under a microscope. The dataset contains measurements of cell and nuclear characteristics, classifying each instance as benign or malignant.

#### Fine Needle Aspiration Biopsy (FNA) extraction technique:

This procedure uses a thin needle and suction to extract a small tissue sample, which is then analyzed under a microscope. FNA is commonly used for diagnosing abnormalities in tissues like the thyroid, breast, and lungs. It is effective for examining individual cell characteristics, though not for assessing cell relationships in tissue. 

Fine Needle Aspiration Biopsy (FNA) **excels in identifying** features that **help distinguish between benign and malignant tumors** based on **cell nucleus properties** like size, shape, texture, and symmetry.

#### Key Nuclear Characteristics of cells

* Size: Cancerous cells often have larger nuclei. Measurements include radius, perimeter, and area.
* Shape: Irregular shapes are common in malignant cells.
* Texture: Variation in grayscale values within the nucleus may indicate malignancy.
* Symmetry: Cancerous nuclei tend to be less symmetrical.
* Fractal Dimension: Measures the complexity of the nuclear contour, which can signal malignancy.

This dataset's nuclear features are critical for training machine learning models to classify tumors as benign or malignant. They allow machine learning models to distinguish between benign and malignant tumors, though the small tissue sample can occasionally lead to false negatives. The main objective in this analysis and training is to see which classification models best suit the medical data in the dataset and minimize, detecting a maximum of positive real cancer cases. Avoid getting false negatives is indeed a priority, considering the psychological effects of false positives in cancer detection versus the possibly letal physical effects (and also psychological) of a false negative diagnosis.

Sources: National Cancer Institute (NCI), American Cancer Society, MedlinePlus.

## **Description**

This dataset contains information about breast cancer cases, with features calculated from digitized images of fine needle aspirate (FNA) of breast masses. It is used for classification tasks in health and medicine, specifically to distinguish between malignant and benign tumors.

- **Type of Data**: Multivariate
- **Field of Study**: Health and Medicine
- **Associated Tasks**: Classification
- **Shape**: 569 rows, 32 columns (including the target variable)
- **Variables**: 30 numerical features, 1 ID column, 1 target column (diagnosis)
- **Missing Values**: No
- **Source**: UCI Machine Learning Repository
- **URL**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Year of Collection**: 1995
- **Memory Usage**: Approximately 1.5 MB
- **File Size**: ~50 KB (CSV format)
- **Authors/Entity**: Dr. William H. Wolberg, University of Wisconsin

### Features

1. **ID**: Unique identifier for each case (Categorical)
2. **Diagnosis**: Diagnosis of the sample (M = malignant, B = benign) (Categorical)
3. **radius_mean**: Mean of distances from center to points on the perimeter (Continuous)
4. **texture_mean**: Standard deviation of gray-scale values (Continuous)
5. **perimeter_mean**: Mean size of the tumor (Continuous)
6. **area_mean**: Area of the tumor (Continuous)
7. **smoothness_mean**: Local variation in radius lengths (Continuous)
8. **compactness_mean**: (Perimeter^2 / Area - 1.0) (Continuous)
9. **concavity_mean**: Severity of concave portions of the contour (Continuous)
10. **concave points_mean**: Number of concave portions of the contour (Continuous)
11. **symmetry_mean**: Symmetry of the tumor (Continuous)
12. **fractal_dimension_mean**: "Coastline approximation" - 1 (Continuous)
13. **radius_se**: Standard error of the radius (Continuous)
14. **texture_se**: Standard error of the texture (Continuous)
15. **perimeter_se**: Standard error of the perimeter (Continuous)
16. **area_se**: Standard error of the area (Continuous)
17. **smoothness_se**: Standard error of the smoothness (Continuous)
18. **compactness_se**: Standard error of the compactness (Continuous)
19. **concavity_se**: Standard error of the concavity (Continuous)
20. **concave points_se**: Standard error of the concave points (Continuous)
21. **symmetry_se**: Standard error of the symmetry (Continuous)
22. **fractal_dimension_se**: Standard error of the fractal dimension (Continuous)
23. **radius_worst**: Worst or largest mean value for the radius (Continuous)
24. **texture_worst**: Worst or largest mean value for the texture (Continuous)
25. **perimeter_worst**: Worst or largest mean value for the perimeter (Continuous)
26. **area_worst**: Worst or largest mean value for the area (Continuous)
27. **smoothness_worst**: Worst or largest mean value for the smoothness (Continuous)
28. **compactness_worst**: Worst or largest mean value for the compactness (Continuous)
29. **concavity_worst**: Worst or largest mean value for the concavity (Continuous)
30. **concave points_worst**: Worst or largest mean value for the concave points (Continuous)
31. **symmetry_worst**: Worst or largest mean value for the symmetry (Continuous)
32. **fractal_dimension_worst**: Worst or largest mean value for the fractal dimension (Continuous)

### Additional Information
- **Targtet: "Diagnosis"**
- **No missing values**.
- For more details and to download the dataset, visit the UCI Machine Learning Repository.

In the dataset, cell nucleus characteristics are presented in three versions: mean, standard error (SE), and worst (largest). This results in 30 feature columns for each nucleus, derived from 10 base features, each recorded in these three versions:

* **Mean** (e.g., ‘radius_mean’):
  - mean values show general trends
  - Represents average values, giving a general idea of cell characteristics. These are the average of measurements taken from all cell nuclei in the sample, providing a central tendency.

* **Standard Error (SE)** (e.g., ‘radius_se’):
  - SE values indicate the reliability of these trends
  - Indicates how much the sample mean would vary if different samples were taken.
  - Lower SE suggests more reliable average values, measuring variability or uncertainty in the average characteristics.

* **Worst** (e.g., ‘radius_worst’):
  - worst values highlight potential outliers or extreme cases indicative of malignancy.
  - Highlights the most extreme measurements, crucial for identifying the most abnormal cells, often key in cancer detection.
  - These are the largest measurements (average of the three largest) for each characteristic, capturing the most extreme values typical of malignant cells.

Having mean, SE, and worst values provides a **comprehensive representation of each feature**, helping to understand **central tendency, variability, and extreme cases**. Different aspects of the data can provide critical diagnostic information. Including these different measures allows for better feature engineering and model building. Machine learning models can use these varied perspectives to improve classification accuracy.

### Files to store and load once dataset is downloaded:

- wdbc.data: Contains the dataset with features and labels.
- wdbc.names: Contains metadata about the dataset, including variable descriptions and data collection methods.

### References
- K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34.


## **Machine Learning**

We will experiment with nine Machine Learning models in this the breast cancer detection project. These classification models will be trained and tested **firstly on the whole dataset of 32 variables** (including ID + Diagnosis as the target variable) and afterwards, we'll do the same on a **reduced dataset using only 17 of the variables**, which were based on the **"features importance"** calculus matched with **correlations results**. 

### Traing models with **PySpark**:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Naive Bayes
* Multilayer Perceptron
* Linear SVC

### Traing models with **Scikit-Learn** (given that Spark does not load these two models):

* XGBClassifier
* Support Vector Machines (SVM). 

### Evaluation metrics valued and discussed

We evaluate checking how each classification model gives best resuults and comment how reducing the number of features to train has affected the results. The metrics analysis is meant to optimize model sensitivity and minimizing false negatives.

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

## **Results of performance metrics for each Classification Model:**

- **Logistic Regression**
  - Complete Set (30 variables): Precision 0.938, Recall 1.000, F1-Score 0.968. Excellent performance, no false negatives, and very few false positives.
  - Reduced Set (13 variables): Precision 0.857, Recall 1.000, F1-Score 0.923. Maintains perfect Recall, though precision decreases, indicating more false positives.
  - Explanation: Logistic Regression remains robust in both datasets, though it loses some precision with the reduction of variables due to the loss of discriminative features.

- **Decision Tree**
  - Complete Set (30 variables): Precision 0.848, Recall 0.933, F1-Score 0.889. Performance: Good, but with some false negatives and positives.
  - Reduced Set (13 variables): Precision 0.879, Recall 0.967, F1-Score 0.921. Slight improvement in precision and Recall, possibly due to the elimination of noise and less significant variables.
  - Explanation: Decision Tree can benefit from the reduction of variables if it removes irrelevant features that cause overfitting.

- **Random Forest**
  - Complete Set (30 variables): Precision 0.879, Recall 0.967, F1-Score 0.921. Performance: Solid, with few false negatives and positives.
  - Reduced Set (13 variables): Precision 0.879, Recall 0.967, F1-Score 0.921. Consistent, maintains similar metrics.
  - Explanation: Random Forest is robust to variable reduction due to its aggregation nature, which can handle smaller feature sets without losing precision.

- **Gradient Boosting**
  - Complete Set (30 variables): Precision 0.857, Recall 1.000, F1-Score 0.923. Performance: Very good, no false negatives, and some false positives.
  - Reduced Set (13 variables): Precision 0.853, Recall 0.967, F1-Score 0.906. Slight decrease in Recall and precision, but still robust.
  - Explanation: Gradient Boosting handles variable reduction well, though a slight loss in Recall indicates that some important features were removed.

- **Naive Bayes**
  - Complete Set (30 variables): Precision 0.800, Recall 0.800, F1-Score 0.800. Performance: The worst performance with a high rate of false negatives.
  - Reduced Set (13 variables): Precision 0.885, Recall 0.767, F1-Score 0.821. Increases in precision, but still has low Recall with significant false negatives and positives.
  - Explanation: Naive Bayes does not handle non-normal distributions and feature correlations well, explaining its poor performance in both datasets.

- **Multilayer Perceptron**
  - Complete Set (30 variables): Precision 0.968, Recall 1.000, F1-Score 0.984. Performance: Excellent, with near-perfect precision and Recall.
  - Reduced Set (13 variables): Precision 0.829, Recall 0.967, F1-Score 0.892. Decreases in precision and Recall, though still solid.
  - Explanation: Variable reduction affects MLP more due to its reliance on multiple features to detect complex patterns.

- **Linear SVC**
  - Complete Set (30 variables): Precision 0.968, Recall 1.000, F1-Score 0.984. Performance: Excellent, with perfect metrics.
  - Reduced Set (13 variables): Precision 0.853, Recall 0.967, F1-Score 0.906. Slight decrease in precision and Recall, but still robust.
  - Explanation: Linear SVC adapts well to variable reduction due to its ability to generalize with a smaller feature set.

- **XGBClassifier**
  - Complete Set (30 variables): Precision 0.956, Recall 1.000, F1-Score 0.977. Performance: Excellent.
  - Reduced Set (13 variables): Precision 0.927, Recall 0.884, F1-Score 0.905. Notable decrease in Recall and precision.
  - Explanation: Variable reduction significantly affects XGBClassifier, possibly because it relies on a broader set of features for optimal performance.

- **Support Vector Machines (SVM)**
  - Complete Set (30 variables): Precision 1.000, Recall 0.930, F1-Score 0.964. Performance: Very good, with perfect precision and slight impact on Recall.
  - Reduced Set (13 variables): Precision 1.000, Recall 0.884, F1-Score 0.938. Maintains high precision, with a slight decrease in Recall.
  - Explanation: SVM shows robustness to variable reduction, possibly due to its ability to maximize decision margins with fewer features.

**Best Models:**
- Linear SVC, Logistic Regression, and Support Vector Machines (SVM)
- These models maintain high levels of precision and Recall even with variable reduction, making them ideal for detecting malignant tumors.

**Less Efficient Model:**
- Naive Bayes, which consistently shows the worst performance due to its inability to handle non-normal distributions and feature correlations.

**Practical Decision:** 
- To detect malignant tumors with this dataset, we recommended using Linear SVC, Logistic Regression, Support Vector Machines, or Multilayer Perceptron, as they are excellent overall when working with both the complete and reduced datasets. They offer a relative balance between precision and Recall, minimizing both false negatives and false positives. 

- It is recommended to use more than one model and, if possible, those that work differently, to be able to contrast the results of one and the other and proceed to clinical studies in case of doubt.
