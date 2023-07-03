from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from joblib import load
def models():
    PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
    train_data_file = 'train.csv'
    train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)
# Read data
    df = pd.read_csv(train_data_path, sep=",")
    X = df.drop('risk', axis=1)
    y = df['risk']

    # Set the number of folds for cross-validation
    k = 5

    # Create a KFold object
    kf = KFold(n_splits=k, shuffle=True)

    # Initialize a list to store the accuracy scores for each fold
    accuracy_scores = []

    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = SVC()
        lr=SVC()
        lr.fit(X_train,y_train)
        from joblib import dump
        dump(lr, 'lr.joblib')
        # Train the model on the training data
        

        # Evaluate the model on the test data
        accuracy = lr.score(X_test, y_test)

        # Store the accuracy score for the current fold
        accuracy_scores.append(accuracy)

        #print(plot_confusion_matrix(model,X_test,y_test))

    # Compute the average accuracy across all folds
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    # Print the average accuracy
    print("Average accuracy:", average_accuracy)
if __name__ == '__main__':
    models()