# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

# Load the Titanic dataset
titanic_data = pd.read_csv('D:\\Projects\\try_2\\titanic.csv')  # You need to replace 'titanic.csv' with your dataset

# Data preprocessing
# For simplicity, let's use a subset of features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = titanic_data[features]
y = titanic_data['Survived']

# Handle missing values and convert categorical variables to numerical
X['Age'].fillna(X['Age'].median(), inplace=True)
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Flask web application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])

        # Make a prediction
        input_data = [[pclass, sex, age, sibsp, parch]]
        prediction_value = model.predict(input_data)
        print(prediction_value)

        return render_template('result.html', prediction=prediction_value)

if __name__ == '__main__':
    app.run(debug=True)
