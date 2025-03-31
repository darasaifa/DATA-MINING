import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=";")

    print(df.info())

    categorical_columns = ['job', 'marital', 'education', 'default', 
                           'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    all_features = numerical_columns + categorical_columns
    df = df[all_features]

    return df, label_encoders

if __name__ == "__main__":
    file_path = r"C:\Users\LENOVO\Documents\Semester 4\Data Mining\WEEK6\MarketingTarget.csv"
    df, _ = load_and_preprocess_data(file_path)
    print(df.head()) 