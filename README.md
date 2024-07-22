# AI Assisted Brain Tumor Localization and Classification in Medical Imaging

This project focuses on classifying MRI images to determine whether they have tumors and segmenting the tumor regions using advanced deep learning techniques. The dataset contains MRI images with corresponding masks indicating tumor regions. We utilize ResNet for classification and ResUNet for segmentation tasks. The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Dataset

The dataset consists of MRI images labeled with masks indicating tumor regions. These images are used to train and evaluate the deep learning models. You can find more details about the dataset [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Project Structure

1. **Data Loading and Preprocessing**
    - Loading the dataset
    - Preprocessing the images
    
2. **Model Building**
    - Splitting the data into training and testing sets
    - Defining the deep learning model architectures
    - Compiling the models
    - Training the models
    - Hyperparameter tuning

3. **Model Evaluation**
    - Evaluating model performance
    - Visualizing results
    - Performance comparison of different models

## Data Loading and Preprocessing

### Data Loading and Preprocessing

Here, we load the dataset and inspect it.

#### Visualizations of MRI Scans and Masks

#### Distribution of Labels in Dataset

![balance](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/c9c02921-25a7-41b6-8232-3178512d0104)

#### Visualizing the images (MRI and Mask) in the dataset separately
![Visualizing the images (MRI and Mask) in the dataset separately](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/f6f3621f-f9ba-42da-abd6-f9e8f09e7863)

#### Visualization of MRI Scans and Corresponding Masks for Sick Patients
![Visualizing 6 randomly selected (1) MRI scan images from only sick patients followed by (2) corresponding mask, (3) both MRI image and the corresponding mask (in red color) on top of each other](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/514bd9de-63c0-4777-a495-54934f1bea14)


### Splitting the Dataset into Training, Validation, and Testing Sets

The dataset is split into training, validation, and testing sets to evaluate the model performance effectively.

### Preprocessing the Images

The images are resized and normalized to prepare them for model training.

## Model Building

### Defining the Model Architecture

Deep learning models are defined using ResNet for classification and ResUNet for segmentation tasks.

### Models Used

We experimented with the following models:
- **ResNet**: For classification tasks to detect the presence of a tumor in MRI images.
- **ResUNet**: For segmentation tasks to identify the tumor region in MRI images.

### Compiling the Models

The models are compiled using appropriate loss functions and optimizers.

### Training the Models

Both models are trained on the training data from scratch.

## Model Evaluation 

### Evaluating Model Performance

The models are evaluated using metrics such as accuracy, precision, recall, and F1 score.

#### Results of ResNet on Test Data

##### Confusion Matrix
![Confusion Matrix](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/76624873-b3b8-48d2-9e76-b07d6a3da878)

##### Matrices With Scores
![Matrices With Scores](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/df68bdb1-876d-454a-a630-f2c470c311a7)

The  **ResUNet** model performed with an accuracy of **0.98** and an F1 Score of **0.98**.

### Visualizing Results

#### Visualization of MRI Scans with Original and AI Predicted Masks
![Visualization of MRI Scans with Original and AI Predicted Masks](https://github.com/QHaider4622/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging/assets/79516393/894ba023-7990-4e1f-8f18-898bf5b72c15)

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets)
- [Excellent Resource on transfer learning by Dipanjan Sarkar](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
- [Article by Jason Brownlee on transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
- [Link to ResNet paper entitled (Deep Residual Learning for Image Recognition)](https://arxiv.org/pdf/1512.03385.pdf)
- Resunet Resouces
  - [Paper 1](https://arxiv.org/abs/1505.04597)
  - [Paper 2](https://arxiv.org/abs/1904.00592)
  - [Great Article](https://aditi-mittal.medium.com/introduction-to-u-net-and-res-net-for-image-segmentation-9afcb432ee2f)
- [Link to Weights](https://drive.google.com/drive/folders/12AgjyExAPuSQ6WmvQUMaGmpS2m4wjb1w?usp=drive_link)

  <!--
