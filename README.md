# AI Assisted Brain Tumor Localization and Classification in Medical Imaging

This project focuses on classifying MRI images to determine whether they have tumors and segmenting the tumor regions using advanced deep learning techniques. The dataset contains MRI images with corresponding masks indicating tumor regions. We utilize ResNet for classification and ResUNet for segmentation tasks. The dataset used in this project is from [Kaggle]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

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


The  **ResUNet** model performed with an accuracy of ** ** and an F1 Score of ** **.

### Visualizing Results

Visualizations are created to understand the models' performance.



## References
- [Kaggle Dataset](https://www.kaggle.com/datasets)
