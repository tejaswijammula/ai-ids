
from data_loader import load_nsl_kdd_data
from preprocessing import preprocess_data
from models.random_forest import train_random_forest
from evaluation import evaluate_model
from utils import plot_confusion

def main():
    train_path = "path_to_train.csv"
    test_path = "path_to_test.csv"

    train_df, test_df = load_nsl_kdd_data(train_path, test_path)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
    plot_confusion(y_test, y_pred_rf, title="Random Forest Confusion Matrix")

if __name__ == "__main__":
    main()
