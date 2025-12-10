# DEEP-LEARNING-PROJECT

*COMPANY*: COTECH IT SOLUTIONS

*NAME*: R.S.PRIYADHARSHINI

*INTERN ID*: CTO4DR2648

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## DEEP LEARNING PROJECT DESCRIPTION

This project focuses on building a deep learning model to classify handwritten digits from 0 to 9 using the MNIST dataset. The MNIST dataset is one of the most popular datasets in machine learning because it is simple, clean, and widely used for practicing image classification. It consists of 70,000 grayscale images of digits: 60,000 for training and 10,000 for testing. Each image is 28×28 pixels and contains a single handwritten digit written by different people. The goal of this project is to develop a Convolutional Neural Network (CNN) using TensorFlow that can correctly identify these digits with high accuracy.

1. Understanding the Dataset

Before building the model, the dataset is loaded directly from TensorFlow’s built-in library. Since CNNs work better when input values are scaled, each pixel value (0 to 255) is divided by 255.0 to normalize the data. This ensures the model trains faster and avoids unnecessary errors. The images are reshaped to include a single color channel, because CNN layers expect a 4D input: (samples, height, width, channels).

2. Building the Convolutional Neural Network

A CNN is chosen because it is the most efficient model for image-based tasks. It automatically learns shapes, edges, curves, and patterns from images without requiring manual feature extraction.

The CNN architecture used here includes:

✔ First Convolution Layer (32 filters, 3×3 kernel)

This layer scans the image and captures low-level features like edges.

✔ MaxPooling Layer (2×2)

This reduces image size and removes unnecessary details while keeping important features.

✔ Second Convolution + Pooling

This time, 64 filters are used, allowing the network to learn more complex patterns such as loops, curves, and handwriting style of different digits.

✔ Flatten Layer

Converts the 2D feature maps into a 1D vector for feeding into dense layers.

✔ Dense Layer (128 units)

A fully connected layer that learns deeper relationships between features.

✔ Output Layer (10 units with softmax)

The final layer predicts probabilities for digits 0 to 9.

The model uses Adam optimizer, a widely used algorithm that adjusts learning speed, and sparse categorical crossentropy as the loss function because this is a multi-class classification problem.

3. Training the Model

The model is trained for 5 epochs. An epoch means the CNN has seen the entire training dataset once. During training, validation data (10% of the training set) is used to check how well the model learns. The accuracy gradually increases while loss decreases, indicating that the model is properly learning the patterns.

4. Model Evaluation

After training, the model is tested on the 10,000 images it has never seen before. The model achieves a very high accuracy of around 98%, which proves that CNNs handle handwritten digit classification extremely well. This performance is typical for MNIST and shows that the model is correctly implemented.

5. Visualization of Results

To understand training behavior, two graphs are plotted:

✔ Accuracy Curve

This shows training accuracy and validation accuracy across epochs.

✔ Loss Curve

Shows how training and validation loss decrease over time.

Both graphs help confirm whether the model is learning correctly or overfitting.

Additionally, the first five test images are displayed along with the model predictions. The predicted digit is shown above each image, making the project visually complete.

6. Conclusion

This project successfully demonstrates how to build, train, evaluate, and visualize a deep learning model in TensorFlow. It uses a CNN architecture that learns patterns from handwritten images and predicts digits with high accuracy. The project includes all necessary steps such as data preprocessing, model building, training, evaluation, and visualization. With around 98% accuracy, the model proves to be highly reliable for digit classification. This fulfills all internship requirements and clearly shows understanding of deep learning concepts.

