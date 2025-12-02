Titanic Dataset Analysis and Survival Prediction

This project aims to explore the Titanic dataset and build a predictive model to forecast whether a passenger survived or not based on various features such as age, sex, passenger class, and more. The project uses Logistic Regression for classification and includes a thorough analysis and data visualization.

Table of Contents

Project Overview

Dataset

Installation

Data Cleaning and Preprocessing

Model Building and Evaluation

Data Visualization

Results

Conclusion


Project Overview

The Titanic dataset contains information about passengers who were aboard the Titanic, including whether they survived, their age, class, fare, and more. This project involves data preprocessing, building a Logistic Regression model for survival prediction, and evaluating the model's performance. Additionally, various visualizations are created to better understand the dataset.

Dataset

The dataset used in this project is the Titanic Training Dataset which is available from Kaggle. It includes the following features:

Survived: Target variable (0 = No, 1 = Yes)

Pclass: Passenger class (1, 2, 3)

Name: Name of the passenger

Sex: Gender (male, female)

Age: Age of the passenger

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket: Ticket number

Fare: Fare paid for the ticket

Cabin: Cabin number

Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

PassengerId: Passenger ID

Installation

To run this project, you need to install the following dependencies:

pip install pandas seaborn matplotlib scikit-learn

Data Cleaning and Preprocessing

Before building the model, the dataset goes through several preprocessing steps:

Drop irrelevant columns: Cabin, Name, Ticket, and PassengerId are dropped as they are not useful for the analysis.

Handling missing values:

Missing Age values are filled with the median age of the passengers.

Missing Embarked values are filled with the mode (most frequent value).

Label Encoding: Categorical features like Sex and Embarked are encoded into numeric values using LabelEncoder.

Model Building and Evaluation
Steps:

Feature Selection: The target variable Survived is separated from the features.

Data Splitting: The data is split into training and testing sets (80% training, 20% testing) using train_test_split.

Logistic Regression: A Logistic Regression model is built and trained on the training set.

Prediction and Evaluation: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


Confusion Matrix and Classification Report are displayed to assess the model's performance.

Data Visualization

Several visualizations are created to explore the dataset:

Boxplot: Age vs Passenger Class (Pclass)

Histogram: Distribution of Age

Scatterplot: Age vs Fare with Survival as Hue

Pairplot: Pairwise relationships between Age, Fare, Pclass, and Survival

Violin Plot: Age distribution by Pclass

Heatmap: Correlation matrix of features

Pie Chart: Distribution of passengers by Pclass

Countplot: Survival count by Sex

These visualizations help to understand trends, distributions, and relationships in the data.

Results

The Logistic Regression model achieves a decent accuracy on predicting survival. The evaluation metrics (precision, recall, F1-score) provide insights into the model's ability to correctly classify survivors and non-survivors.

Conclusion

The Titanic dataset is a great starting point for practicing data analysis, preprocessing, and machine learning.

<img width="562" height="455" alt="download" src="https://github.com/user-attachments/assets/94ce4473-1aff-4e4c-a4c5-9af8f94e2139" />
<img width="571" height="455" alt="download (1)" src="https://github.com/user-attachments/assets/d8fbd5d9-aa41-4800-b7b9-e01de93fff17" />
<img width="571" height="455" alt="download (2)" src="https://github.com/user-attachments/assets/245912c0-ba8f-4456-912a-c3074c548a7f" />
<img width="989" height="1023" alt="download (3)" src="https://github.com/user-attachments/assets/af47937f-ddaf-48aa-bc55-824f37c7aa0a" />
<img width="562" height="455" alt="download (4)" src="https://github.com/user-attachments/assets/1e012f07-8c46-484a-82e1-36054aa5d37f" />
<img width="583" height="492" alt="download (5)" src="https://github.com/user-attachments/assets/dceb0271-643c-460d-b434-486f7cb8704b" />
<img width="389" height="411" alt="download (6)" src="https://github.com/user-attachments/assets/8caf921d-dfcd-46b5-8ab9-c3c47b117568" />
<img width="571" height="455" alt="download (7)" src="https://github.com/user-attachments/assets/ea139118-15a0-464d-86c0-c24d723b0a60" />











Logistic Regression was applied for survival prediction, and performance evaluation shows that the model can predict survival with reasonable accuracy.

Data visualization helped to uncover useful patterns and trends in the dataset, contributing to a better understanding of the relationships between features.
