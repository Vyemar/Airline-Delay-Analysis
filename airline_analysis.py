# ---------------- Libraries ----------------
from matplotlib import cm
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# ---------------- Global ----------------
num_of_rows = 100000


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
    def clean_one(df):
        if df is None:
            return None

        df = df.copy()

        # Treat NA and blank strings as missing values
        df = df.replace(["NA", "", " "], np.nan)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Convert columns that should be numeric
        numeric_cols = [
            "ActualElapsedTime", "AirTime", "ArrDelay", "ArrTime", "CRSArrTime",
            "CRSDepTime", "CRSElapsedTime", "Cancelled", "CarrierDelay",
            "DayOfWeek", "DayofMonth", "DepDelay", "DepTime", "Distance",
            "Diverted", "FlightNum", "LateAircraftDelay", "Month", "NASDelay",
            "SecurityDelay", "TaxiIn", "TaxiOut", "WeatherDelay", "Year"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove cancelled and diverted flights
        if "Cancelled" in df.columns:
            df = df[df["Cancelled"] != 1]

        if "Diverted" in df.columns:
            df = df[df["Diverted"] != 1]

        # Remove rows missing target variable
        if "ArrDelay" in df.columns:
            df = df.dropna(subset=["ArrDelay"])

        # Remove clearly impossible values
        if "Distance" in df.columns:
            df = df[df["Distance"] >= 0]

        for col in ["ActualElapsedTime", "AirTime", "CRSElapsedTime", "TaxiIn", "TaxiOut"]:
            if col in df.columns:
                df = df[(df[col].isna()) | (df[col] >= 0)]

        # These columns leak the answer or are post-arrival delay breakdowns
        leakage_cols = [
            #"ArrDelay", Had to remove this one to prevent crashes
            "CarrierDelay",
            "LateAircraftDelay",
            "NASDelay",
            "SecurityDelay",
            "WeatherDelay",
            "CancellationCode"
        ]

        existing_leakage = [col for col in leakage_cols if col in df.columns]
        df = df.drop(columns=existing_leakage)

        # Fill missing numeric columns with median
        remaining_numeric = df.select_dtypes(include=[np.number]).columns
        for col in remaining_numeric:
            df[col] = df[col].fillna(df[col].median())

        # Fill missing categorical columns with mode
        remaining_object = df.select_dtypes(include=["object"]).columns
        for col in remaining_object:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")

        return df.reset_index(drop=True)

    testSet = clean_one(testSet)
    trainSet = clean_one(trainSet)
    validateSet = clean_one(validateSet)

    return testSet, trainSet, validateSet


# Function to choose relevant features
'''
This function chooses the most important features for input and analysis.
It returns the selected features.

Important features include times, delays, airport/airline identifiers, and operational factors.

- Meg
'''
def relevant_features(testSet, trainSet, validateSet):
    selected_columns = [
        "Year",
        "Month",
        "DayOfWeek",
        "DayofMonth",
        "CRSDepTime",
        "DepTime",
        "DepDelay",
        "CRSArrTime",
        "CRSElapsedTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "TaxiOut",
        "FlightNum",
        "Distance",
        "UniqueCarrier",
        "Origin",
        "Dest",
        "TailNum"
    ]

    def select_cols(df):
        available_cols = [col for col in selected_columns if col in df.columns]
        return df[available_cols].copy()

    testX = select_cols(testSet)
    trainX = select_cols(trainSet)
    validateX = select_cols(validateSet)

    return testX, trainX, validateX


# Function to encode categorical features
'''
This function converts non-numeric categorical features into numeric format.

It returns the transformed features.

- Meg
'''
def encode_features(testX, trainX, validateX):
    # Combine first so all sets get the same variable columns
    combined = pd.concat(
        [trainX, validateX, testX],
        keys=["train", "validate", "test"]
    )

    categorical_cols = ["UniqueCarrier", "Origin", "Dest", "TailNum"]
    categorical_cols = [col for col in categorical_cols if col in combined.columns]

    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    trainX = combined.xs("train")
    validateX = combined.xs("validate")
    testX = combined.xs("test")

    return testX, trainX, validateX


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
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    return model



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
def validate_model(model, xValidate, yValidate):
    predictions = model.predict(xValidate)
    accuracy = accuracy_score(yValidate, predictions)
    precision = precision_score(yValidate, predictions, zero_division=0)
    recall = recall_score(yValidate, predictions, zero_division=0)
    f1 = f1_score(yValidate, predictions, zero_division=0)
    matrix = confusion_matrix(yValidate, predictions)

    print("Validation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }


# Function to predict delays
'''
This function uses the trained decision tree to predict if flights will be delayed.

It returns 1 for a predicted delay and 0 for a predicted on-time arrival.

- Lency
'''
def predict_delays(model, X_test):
    try:
        predictions = model.predict(X_test)
        print("Predictions completed.")
        return predictions
    except Exception as e:
        print(f"Something went wrong while predicting: {e}")
        return None


# Function to analyze feature importance
'''
This function identifies which features were most important in the decision tree.

Returns feature importance results.

- Meg
'''
def feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("Top Feature Importances:")
    print(importance_df.head(15))

    return importance_df


# Main function
'''
Initialized code for airline analysis.
'''
if __name__ == "__main__":
    dataFile = load_data()
    testSet, trainSet, validateSet = split_data(dataFile)
    testSet, trainSet, validateSet = clean_data(testSet, trainSet, validateSet)
    yTrain = target_variable(trainSet)
    yValidate = target_variable(validateSet)
    yTest = target_variable(testSet)
    testX, trainX, validateX = relevant_features(testSet, trainSet, validateSet)
    testX, trainX, validateX = encode_features(testX, trainX, validateX)
    model = build_model()
    trained_model = train_model(model, trainX, yTrain)
    validate_model(trained_model, validateX, yValidate)
    test_predictions = predict_delays(trained_model, testX)
    feature_importance(trained_model, trainX.columns)