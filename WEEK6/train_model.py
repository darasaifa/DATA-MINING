from sklearn.ensemble import RandomForestClassifier
from split_data import split_data, load_and_preprocess_data

def train_model(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    file_path = r"C:\Users\LENOVO\Documents\Semester 4\Data Mining\WEEK6\MarketingTarget.csv"
    df, _ = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
