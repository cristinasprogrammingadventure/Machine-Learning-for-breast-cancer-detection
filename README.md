# Breast cancer detection project

## Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

### Introduction and Context

This dataset comes from the analysis of breast tumors using Fine Needle Aspiration (FNA) biopsy, where tissue samples are observed under a microscope. The dataset contains measurements of cell and nuclear characteristics, classifying each instance as benign or malignant.

#### Fine Needle Aspiration Biopsy (FNA)
This procedure uses a thin needle and suction to extract a small tissue sample, which is then analyzed under a microscope. FNA is commonly used for diagnosing abnormalities in tissues like the thyroid, breast, and lungs. It is effective for examining individual cell characteristics, though not for assessing cell relationships in tissue. As such, FNA excels in identifying features that help distinguish between benign and malignant tumors based on cell nucleus properties like size, shape, texture, and symmetry.

#### Key Nuclear Characteristics:

Nucleus Size: Cancerous cells often have larger nuclei. Measurements include radius, perimeter, and area.
Nucleus Shape: Irregular shapes are common in malignant cells.
Nuclear Texture: Variation in grayscale values within the nucleus may indicate malignancy.
Nuclear Symmetry: Cancerous nuclei tend to be less symmetrical.
Fractal Dimension: Measures the complexity of the nuclear contour, which can signal malignancy.

This dataset's nuclear features are critical for training machine learning models to classify tumors as benign or malignant.

Sources: National Cancer Institute (NCI), American Cancer Society, MedlinePlus.

### Machine Learning 

#### Models 
For the breast cancer detection project, I used classification models such as Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, Multilayer Perceptron, Linear SVC, XGBClassifier, and Support Vector Machines (SVM). 

#### Models' evaluation metrics
Evaluation metrics included Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix, optimizing model sensitivity and minimizing false negatives.
