# ---------------- Libraries ----------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# ---------------- Global ----------------
num_of_rows = 500000


# ---------------- Functions ----------------
# Function to load data
'''
This function loads the airline data from the csv file.
It returns the loaded data.

Function needs to randomly sample 500k-1m rows from original dataset 
to manage computer memory constraints.

Return loaded dataset

- Roxie
'''
def load_data():
    try:
        file = pd.read_csv("airline.csv.shuffle",encoding="latin1",nrows=num_of_rows)
        #Test by printing first 20
        #print(file.head(20))
        print("File successfully loaded.")
        return file
    except FileNotFoundError:
        print("File not found! Please ensure airline.csv.shuffle is in the same directory as this program (airline_analysis.py)")
        return None


# Function to clean data
'''
This function acts to clean the data set.

Rows are removed if any of the following are true:
- Duplicate row
- Cancelled flight (Cancelled == 1)
- Diverted flight (Diverted == 1)
- Missing values in the target variable (Arrival Delay)
- Impossible values (negative delays, departure times after arrival times, etc.)
- Column provides an answer to the question "Was the flight delayed?"

Missing values are filled with the median value of the feature 
across the random sample of the dataset.

Numeric strings are converted to integer values.

Return cleaned dataset

- Meg
'''
def clean_data(testSet, trainSet, validateSet):
    pass


# Function to choose relevant features
'''
This function chooses the most important features for input and analysis.
It returns the selected features.

Important features include times, delays, airport/airline identifiers, and operational factors.

- Meg
'''
def relevant_features():
    pass


# Function to encode categorical features
'''
This function converts non-numeric categorical features into numeric format.

It returns the transformed features.

- Meg
'''
def encode_features():
    pass


# Function to define target variable
'''
This function defines the target variable for analysis (whether or not a flight was delayed).
It should define exactly what the model is predicting based on the cleaned and processed features.

It returns the target variable (0 for a on-time arrival, 1 for a delayed arrival).

- Roxie
'''
def target_variable(df):
    if df is None:
        print("Target variable not found")
        return None
    #If we are 20 minutes late, it's a delay.
    target = (df['ArrDelay'] >= 20).astype(int)
    return target


# Function to split data
'''
This function splits the cleaned data into training and testing sets.
It returns the training and testing datasets.

return training and testing datasets

- Lency
'''
def split_data(data):
    # check if data is loaded before attempting to split
    if data is None:
        print("No data to split. Please load the data first.")
        return None, None, None
    
    # Shuffle the data before splitting to ensure randomness
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    #split points for 70% training, 15% validation, 15% testing
    train_size = int(0.7 * len(shuffled_data))
    validate_end = int(0.85 * len(shuffled_data))

    # Split the data into training, validation, and testing sets
    trainSet = shuffled_data[:train_size]
    validateSet = shuffled_data[train_size:validate_end]
    testSet = shuffled_data[validate_end:]

    print("data successfully split.")
    return testSet, trainSet, validateSet


# Functiopn to build decision tree
'''
This function builds a decision tree for predicting flight delays. It uses gini impurity to
determine the best splits at each node.

It should:
- Create decision tree classifier
- Set tree to use gini impurity
- Define any necessary parameters (like max depth, min samples split, etc.)
- Return the compiled decision tree model ready for training

- Lency
'''
def build_model():
    pass


# Function to train model
'''
This function trains the decision tree using the training data.
It returns the trained model and any relevant training history or metrics.

- Roxie
'''
def train_model(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        print("Model has finished the training phase!")
        return model
    except Exception as e:
        print(f"Something went wrong while training: {e}")
        return None


# Function to validate model
'''
This function validates the decision tree using the testing data.
It returns the validation results, such as accuracy, precision, recall, or other relevant metrics.

- Meg
'''
def validate_model():
    pass


# Function to predict delays
'''
This function uses the trained decision tree to predict if flights will be delayed.

It returns 1 for a predicted delay and 0 for a predicted on-time arrival.

- Lency
'''
def predict_delays():
    pass


# Function to analyze feature importance
'''
This function identifies which features were most important in the decision tree.

Returns feature importance results.

- Meg
'''
def feature_importance():
    pass


# Main function
'''
Initialized code for airline analysis.
'''
if __name__ == "__main__":
    dataFile = load_data()
    testSet, trainSet, validateSet = split_data(dataFile)
    testSet, trainSet, validateSet = clean_data(testSet, trainSet, validateSet)
    y_train = target_variable(trainSet)
    y_validate = target_variable(validateSet)
    tree = build_model()
    delays = ['ArrDelay', 'Cancelled', 'Diverted', 'FlightNum', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    X_train = trainSet.drop(columns=delays).select_dtypes(include=[np.number])
    trained_tree = train_model(tree, X_train, y_train)
