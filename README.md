# Fruit Classification: A Comparative Study of CNN and Random Forest

This project focuses on the classification of four different fruit categories (Apple, Banana, Orange, and Strawberry) using computer vision and deep learning techniques. The study evaluates the performance differences between traditional machine learning methods and modern convolutional neural network (CNN) architectures.

## Project Overview
The primary objective of this research is to analyze how various model architectures handle image data and how data augmentation techniques influence final accuracy.

* **Traditional Approach:** Feature extraction was performed using HOG (Histogram of Oriented Gradients), followed by a Random Forest classifier.
* **Deep Learning Approach:** A custom CNN architecture was developed, consisting of 10 layers and approximately 3.3 million trainable parameters.
* **Optimization:** Techniques such as Dropout, Normalization, and Data Augmentation were implemented to manage overfitting and improve model robustness.

## Performance Results
The models were evaluated on a test dataset, yielding the following accuracy results:

| Model Architecture | Accuracy Rate |
| :--- | :--- |
| **Standard CNN** | **91.11%** |
| Augmented CNN | 84.44% |
| Random Forest (HOG) | 35.56% |

**Analysis:** The Standard CNN achieved the highest performance due to the homogeneous nature of the provided dataset. While Data Augmentation introduced a slight decrease in test accuracy, it enhanced the model's ability to generalize to varied real-world orientations and lighting conditions.

## Technical Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Machine Learning:** Scikit-learn
* **Image Processing:** OpenCV
* **Data Visualization:** Matplotlib / Seaborn

## Repository Structure
* `makinedataset/`: Directory containing training and testing images.
* `Fruit_Classification.ipynb`: Comprehensive Jupyter Notebook containing data preprocessing, model training, and evaluation.
* `models/`: Saved model files in .keras and .joblib formats.

## Conclusion
This study demonstrates that while traditional machine learning provides a baseline, CNN architectures are significantly more effective at capturing spatial hierarchies in image data for classification tasks.
