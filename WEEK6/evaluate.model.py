from sklearn.metrics import accuracy_score
from train_model import train_model
from split_data import split_data, load_and_preprocess_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    file_path = r"C:\Users\LENOVO\Documents\Semester 4\Data Mining\WEEK6\MarketingTarget.csv"
    df, _ = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
