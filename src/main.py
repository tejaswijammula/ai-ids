from data_loader import load_nsl_kdd_data
from preprocessing import preprocess_data
from models.random_forest import train_random_forest
from models.lstm import train_lstm
from models.cnn import train_cnn
from models.ensemble import train_ensemble
from evaluation import evaluate_model
from utils import plot_confusion

def main():
    # Paths to your datasets
    train_path = "path_to_train.csv"
    test_path = "path_to_test.csv"

    # Load and preprocess the data
    train_df, test_df = load_nsl_kdd_data(train_path, test_path)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    # === Train and Evaluate Random Forest ===
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
    plot_confusion(y_test, y_pred_rf, title="Random Forest Confusion Matrix")

    # === Train and Evaluate LSTM ===
    print("\nTraining LSTM...")
    lstm_model = train_lstm(X_train, y_train)
    y_pred_lstm = evaluate_model(lstm_model, X_test, y_test, model_name="LSTM")
    plot_confusion(y_test, y_pred_lstm, title="LSTM Confusion Matrix")

    # === Train and Evaluate CNN ===
    print("\nTraining CNN...")
    cnn_model = train_cnn(X_train, y_train)
    y_pred_cnn = evaluate_model(cnn_model, X_test, y_test, model_name="CNN")
    plot_confusion(y_test, y_pred_cnn, title="CNN Confusion Matrix")

    # === Train and Evaluate Ensemble (Voting) ===
    print("\nTraining Ensemble Model...")
    ensemble_model = train_ensemble([rf_model, lstm_model, cnn_model])
    y_pred_ensemble = evaluate_model(ensemble_model, X_test, y_test, model_name="Ensemble")
    plot_confusion(y_test, y_pred_ensemble, title="Ensemble Confusion Matrix")

if __name__ == "__main__":
    main()
