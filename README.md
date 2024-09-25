# Breast cancer detection project

## Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

### Introduction and Context

This dataset comes from the analysis of breast tumors using Fine Needle Aspiration (FNA) biopsy, where tissue samples are observed under a microscope. The dataset contains measurements of cell and nuclear characteristics, classifying each instance as benign or malignant.

#### Fine Needle Aspiration Biopsy (FNA)
This procedure uses a thin needle and suction to extract a small tissue sample, which is then analyzed under a microscope. FNA is commonly used for diagnosing abnormalities in tissues like the thyroid, breast, and lungs. It is effective for examining individual cell characteristics, though not for assessing cell relationships in tissue. As such, FNA excels in identifying features that help distinguish between benign and malignant tumors based on cell nucleus properties like size, shape, texture, and symmetry.

#### Key Nuclear Characteristics:

* Nucleus Size: Cancerous cells often have larger nuclei. Measurements include radius, perimeter, and area.
* Nucleus Shape: Irregular shapes are common in malignant cells.
* Nuclear Texture: Variation in grayscale values within the nucleus may indicate malignancy.
* Nuclear Symmetry: Cancerous nuclei tend to be less symmetrical.
* Fractal Dimension: Measures the complexity of the nuclear contour, which can signal malignancy.

This dataset's nuclear features are critical for training machine learning models to classify tumors as benign or malignant. They allow machine learning models to distinguish between benign and malignant tumors, though the small tissue sample can occasionally lead to false negatives. The main objective in this analysis and training is to see which classification models best suit the medical data in the dataset and minimize, detecting a maximum of positive real cancer cases. Avoid getting false negatives is indeed a priority, considering the psychological effects of false positives in cancer detection versus the possibly letal physical effects (and also psychological) of a false negative diagnosis.

Sources: National Cancer Institute (NCI), American Cancer Society, MedlinePlus.

### Machine Learning 

#### We will experiment with nine Machine Learning models in this the breast cancer detection project. These classification models will be trained and tested **firstly on the whole dataset of 32 variables (including ID + target) and afterwards, we'll do the same on a reduced dataset using only 17 of the variables, which were chosen with the "features importance" formula matched with correlations. 

Traing models with **PySpark**:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Naive Bayes
* Multilayer Perceptron
* Linear SVC

Traing models with **Scikit-Learn** (given that Spark does not load these two models):

* XGBClassifier
* Support Vector Machines (SVM). 

#### Models' evaluation metrics

Finally, we evaluate checking how each classification model gives best resuults and comment how reducing the number of features to train has affected the results.

Evaluation metrics valued and discussed:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

The metrics analysis is meant to optimize model sensitivity and minimizing false negatives.
