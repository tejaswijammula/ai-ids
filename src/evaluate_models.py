from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from tensorflow.keras.models import Sequential
print(classification_report(y_test, y_pred_rf))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf, labels=[0,1]), display_labels=["Normal", "Attack"]).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred_lstm))

print(classification_report(y_test, y_pred_cnn))

# Step 10: Ensemble (majority voting)
print("Generating ensemble results...")
from scipy.stats import mode
y_pred_ensemble = np.squeeze(mode(np.array([y_pred_rf, y_pred_lstm.flatten(), y_pred_cnn.flatten()]), axis=0).mode)
print("Ensemble Report:")
print(classification_report(y_test, y_pred_ensemble))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_ensemble, labels=[0, 1]), display_labels=["Normal", "Attack"]).plot()
plt.title("Ensemble Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lstm, labels=[0,1]), display_labels=["Normal", "Attack"]).plot()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_cnn, labels=[0,1]), display_labels=["Normal", "Attack"]).plot()