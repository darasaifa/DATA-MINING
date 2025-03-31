from sklearn.model_selection import train_test_split
from load_dataset import load_and_preprocess_data

def split_data(df, test_size=0.2, random_state=42):
   
    X = df.drop(columns=['y'])  
    y = df['y']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = r"C:\Users\LENOVO\Documents\Semester 4\Data Mining\WEEK6\MarketingTarget.csv"
    df, _ = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
