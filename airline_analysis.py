# ---------------- Libraries ----------------


# ---------------- Global ----------------


# ---------------- Functions ----------------
# Function to load data
'''
This function loads the airline data from the csv file.
It returns the loaded data.

Function needs to randomly sample 500k-1m rows from original dataset 
to manage computer memory constraints.

DO NOW - Roxie
'''
def load_data():
    pass


# Function to clean data
'''
This function acts to clean the data set by removing duplicates, handling missing values,
correcting inconsistencies, removing unneccessary columns, and ensuring data types are correct.
It returns the cleaned data set.

DO NOW - Meg
'''
def clean_data():
    pass


# Function to choose relevant features
'''
This function chooses the most important features for input and analysis.
It returns the selected features.

Important features include times, delays, airport/airline identifiers, and operational factors.
'''
def relevant_features():
    pass


# Function to encode categorical features
'''
This function converts non-numeric categorical features into numeric format.
It returns the transformed features.
'''
def encode_features():
    pass


# Function to normalize data
'''
This function normalizes the dataset features to a common scale.
It returns the normalized data and any relevant scaling parameters.
'''
def normalize_data():
    pass


# Function to define target variable
'''
This function defines the target variable for analysis, such as delay status or delay duration.
It should define exactly what the model is predicting based on the cleaned and processed features.
It returns the target variable.
'''
def target_variable():
    pass


# Function to apply pca
'''
This function applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset 
while retaining as much variance as possible.
It returns the PCA-transformed dataset and any relevant PCA parameters.
'''
def apply_pca():
    pass


# Function to choose components
'''
This function determines the optimal number of principal components to retain.
It returns the number of components to retain.
'''
def choose_components():
    pass


# Function to split data
'''
This function splits the cleaned data into training and testing sets.
It returns the training and testing datasets.

DO NOW - Lency
'''
def split_data():
    pass


# Functiopn to build neural network model
'''This function builds a neural network model for predicting flight delays.
It returns the compiled neural network model.'''
def build_model():
    pass


# Function to train model
'''
This function trains the neural network model using the training data.
It returns the trained model and any relevant training history or metrics.
'''
def train_model():
    pass


# Function to validate model
'''
This function validates the neural network model using the testing data.
It returns the validation results, such as accuracy, precision, recall, or other relevant metrics.
'''
def validate_model():
    pass


# Function to calculate conditional probabilities
'''
This function calculates the conditional probabilities of flight delays given certain features or conditions.
It returns the calculated probabilities and any relevant insights derived from them.
'''
def conditional_probabilities():
    pass


# Function to predict delays
'''
This function uses the trained neural network model to predict flight delays on new or unseen data.
It returns the predicted delay status or delay duration for the input data.
'''
def predict_delays():
    pass


# Function to analyze delay correlations
'''
This function analyzes the correlations between different features and flight delays.
It returns insights into which features are most strongly correlated with delays and how they interact with each other
'''
def delay_correlations():
    pass


# Main function
'''
Initialized code for airline analysis.
'''
if __name__ == "__main__":
    pass