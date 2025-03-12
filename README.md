# Netra - Diabetic Retinopathy Detection App

**Netra** is a stage-specific neural network system designed to detect early stages of diabetic retinopathy (DR) using deep learning techniques, specifically a **ResNet50-based Convolutional Neural Network (CNN)**. It helps patients and doctors detect DR by analyzing fundus images. The app integrates data preprocessing techniques and state-of-the-art algorithms to deliver an accurate diagnosis of DR stages. 

## Key Features

- **ResNet50 Architecture**: The deep learning model is based on ResNet50 CNN, optimized for fundus image classification.
- **Early Stage Detection**: The system is trained to detect five different stages of diabetic retinopathy (No DR, Mild, Moderate, Severe, and Proliferative).
- **Fundus Camera Integration**: Integrates with a DIY fundus camera (DIYretCAM) for capturing retina images.
- **Data Preprocessing**: Advanced data preprocessing techniques such as **CLAHE**, **Otsu Thresholding**, **Green Channel Extraction**, and **Circular Cropping** are employed for better feature extraction.
- **Cloud Deployment**: The model is deployed using **Heroku**, and the app interacts with the backend via **Flask API**.

## Technology Stack

- **Deep Learning**: ResNet50-based Convolutional Neural Network.
- **Data Preprocessing**: Contrast-Limited Adaptive Histogram Equalization (CLAHE), Otsu Thresholding, Circular Image Cropping, Green Channel Extraction.
- **Android**: Core app development.
- **Heroku**: Cloud deployment for the trained model.
- **Python**: Flask API for backend communication.

## Mathematical Explanation of Model Preprocessing

### Circular Image Cropping

**Circular Image Cropping** is a technique used to remove uninformative regions (like black backgrounds) from fundus images. This is essential in order to reduce computational load and focus the model on the relevant parts of the retina.

We achieve this by applying a binary mask that selects only the informative circular region of the image, which contains the retina. The process can be represented as follows:

<img src="https://latex.codecogs.com/png.latex?dst(x,%20y)%20%5CRightarrow%20src(x,%20y)%20%5Cquad%20%5Ctext%7Bif%20src(x,y)%20%3E%20threshold,%20%5Ctext%7Botherwise,%20%7D%200%7D" alt="Circular Cropping Equation" />

Where:
- <img src="https://latex.codecogs.com/png.latex?dst(x,%20y)" alt="dst(x, y)" /> represents the resulting pixel value after applying the mask.
- <img src="https://latex.codecogs.com/png.latex?src(x,%20y)" alt="src(x, y)" /> is the original pixel value.
- **threshold** is a predefined value that separates the background from the retina.

The result is a cropped, circular image that removes the background while preserving the retina for analysis.

### Grayscale Conversion

To simplify the image data, **grayscale conversion** is applied, focusing on intensity values rather than color. This transformation is based on the formula:

<img src="https://latex.codecogs.com/png.latex?Y%20%3D%200.299%20%5Ccdot%20R%20+%200.587%20%5Ccdot%20G%20+%200.114%20%5Ccdot%20B" alt="Grayscale Conversion Equation" />

Where:
- <img src="https://latex.codecogs.com/png.latex?R" alt="R" />, <img src="https://latex.codecogs.com/png.latex?G" alt="G" />, and <img src="https://latex.codecogs.com/png.latex?B" alt="B" /> are the red, green, and blue channels of the original image.
- <img src="https://latex.codecogs.com/png.latex?Y" alt="Y" /> is the resulting grayscale pixel intensity.

In diabetic retinopathy detection, the green channel is often used preferentially as it provides the best contrast for the veins and lesions visible in fundus images.

### Contrast-Limited Adaptive Histogram Equalization (CLAHE)

**CLAHE** is a contrast enhancement technique that redistributes pixel intensities in such a way that local contrasts are improved, particularly in regions with poor lighting or contrast. The process is especially useful for enhancing the visibility of small veins or lesions in fundus images.

The cumulative distribution function (CDF) for each pixel intensity \(i\) is calculated as:

<img src="https://latex.codecogs.com/png.latex?cdf_x(i)%20%3D%20%5Csum_%7Bj%3D0%7D%5E%7Bi%7D%20P_x(x%20%3D%20j)" alt="CDF Equation" />

Where:
- <img src="https://latex.codecogs.com/png.latex?cdf_x(i)" alt="cdf_x(i)" /> is the cumulative distribution function for intensity \(i\).
- <img src="https://latex.codecogs.com/png.latex?P_x(x%20%3D%20j)" alt="P_x(x = j)" /> is the probability of intensity \(j\).

After obtaining the CDF, the transformation applied to the grayscale image is given by:

<img src="https://latex.codecogs.com/png.latex?y%20%3D%20T(k)%20%3D%20cdf_x(k)" alt="Transformation Equation" />

Finally, the result is mapped back into the original intensity range:

<img src="https://latex.codecogs.com/png.latex?y%27%20%3D%20%5Ctext%7Bround%7D(y%20%5Ccdot%20(L%20-%201))" alt="Mapping Back Equation" />

Where <img src="https://latex.codecogs.com/png.latex?L" alt="L" /> is the number of gray levels (usually 256 for 8-bit images). This step ensures that the enhanced image fits into the original intensity range.

### Otsu Thresholding

**Otsu Thresholding** is an adaptive thresholding technique used to automatically binarize images by choosing an optimal threshold value. It is particularly effective in distinguishing foreground (e.g., veins or lesions) from the background in fundus images.

Otsu’s method minimizes the weighted within-class variance, which is calculated as:

<img src="https://latex.codecogs.com/png.latex?%5Csigma%5E2_w(t)%20%3D%20q_1(t)%20%5Csigma%5E2_1(t)%20+%20q_2(t)%20%5Csigma%5E2_2(t)" alt="Otsu Equation" />

Where:
- <img src="https://latex.codecogs.com/png.latex?q_1(t)" alt="q1(t)" /> and <img src="https://latex.codecogs.com/png.latex?q_2(t)" alt="q2(t)" /> are the probabilities of the two classes (background and foreground).
- <img src="https://latex.codecogs.com/png.latex?%5Csigma%5E2_1(t)" alt="sigma1(t)" /> and <img src="https://latex.codecogs.com/png.latex?%5Csigma%5E2_2(t)" alt="sigma2(t)" /> are the variances of these classes.

The goal is to find the threshold \(t\) that minimizes \(\sigma^2_w(t)\), thereby maximizing the separability between the two classes. This threshold is then used to binarize the image, effectively highlighting the regions of interest, such as veins or lesions, in fundus images.

### Multinomial Naive Bayes

For the classification task, **Multinomial Naive Bayes** is applied to analyze the extracted features from the preprocessed fundus images and predict the stage of diabetic retinopathy. The Naive Bayes classifier assumes that all features are conditionally independent given the class, and it calculates the posterior probability for each class using Bayes' theorem.

<img src="https://latex.codecogs.com/png.latex?P(C_k%20%7C%20X)%20%3D%20%5Cfrac%7BP(C_k)%20%5Ccdot%20P(X%20%7C%20C_k)%7D%7BP(X)%7D" alt="Bayes Theorem Equation" />

Where:
- <img src="https://latex.codecogs.com/png.latex?P(C_k%20%7C%20X)" alt="P(C_k | X)" /> is the posterior probability of class \(C_k\) (diabetic retinopathy stage) given the feature vector \(X\).
- <img src="https://latex.codecogs.com/png.latex?P(C_k)" alt="P(C_k)" /> is the prior probability of class \(C_k\).
- <img src="https://latex.codecogs.com/png.latex?P(X%20%7C%20C_k)" alt="P(X | C_k)" /> is the likelihood of observing the feature vector \(X\) given the class \(C_k\).
- <img src="https://latex.codecogs.com/png.latex?P(X)" alt="P(X)" /> is the probability of the feature vector \(X\).

## Model Architecture

The deep learning model is based on **ResNet50**, which is optimized to handle high-resolution fundus images. The model was trained using a dataset of over 35,000 fundus images with five different DR stages.

**Layer Summary**:
- Convolutional layers for feature extraction.
- Max pooling layers to reduce dimensionality.
- Batch normalization for better convergence.
- Dense layers for classification into the 5 stages of DR.

**Sample Output**:

| Stage           | Accuracy |
|-----------------|----------|
| No DR           | 95%      |
| Mild            | 93%      |
| Moderate        | 90%      |
| Severe          | 87%      |
| Proliferative   | 85%      |

## Dataset

The model was trained using a dataset provided by **Kaggle**, along with public datasets like **MESSIDOR** and **E-Ophtha**. The dataset includes fundus images that were classified into five stages of diabetic retinopathy.

### Preprocessing

- **Up-sampling**: Used for minority classes to balance the dataset.
- **Down-sampling**: Performed for majority classes to reduce bias.
- **Image Resizing**: Images were resized to 512x512 pixels to reduce the computational load.

## Setup and Installation

### Prerequisites

- **Android Studio**: Required for app development and testing.
- **Heroku Account**: For cloud deployment of the model.
- **Flask**: Backend API setup.
- **Python 3.x**: Required for the machine learning pipeline.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ansschh/Netra-Diabetic_Retinopathy_Detection-App.git
    cd Netra-Diabetic_Retinopathy_Detection-App
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up **Heroku** for model deployment:
    - Deploy the trained ResNet50 model on Heroku using Flask as the backend API.
  
4. Set up the Android app in **Android Studio**:
    - Open the project and sync all dependencies.
  
### Usage

1. **Image Processing**:
    - Capture a 30-second video of the patient’s retina using the **DIYretCAM**.
    - The app extracts individual frames and passes them through the ResNet50 model for classification.

2. **Real-Time DR Detection**:
    - The app displays results for each test case, indicating the DR stage detected for each frame.

## Screenshots

- Home Screen: User uploads the fundus video.
- Result Screen: The app displays the classification of the test cases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Paper and Research

For detailed insights into the development of **Netra** and the underlying research, please refer to the paper:

[Netra: Stage Specific Neural Network for Early Diabetic Retinopathy Detection Using CNN (ResNet50)](./NetraLast.pdf)

## Contact

For any questions or issues, please contact:

- **Ansh Tiwari** (anshtiwari9899@gmail.com)
