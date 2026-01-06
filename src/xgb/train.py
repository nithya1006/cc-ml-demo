from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.dataprep.data_loader import load_housing_data
from src.dataprep.preprocessing import create_preprocessor

def main():
    # Load the dataset
    df = load_housing_data()

    # Split features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Build the preprocessor
    preprocessor = create_preprocessor(X)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the full pipeline (preprocessing + model)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
    ])
    
    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluation
    print("XGBoost Regressor")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    main()