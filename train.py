import torch
from torch import nn, optim
import data_handler
import model as model_arch
import engine
import utils

# --- CONFIGURATION ---
DATA_PATH = "NAB_dataset.csv" # Replace with your dataset path
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
SAVE_PATH = "best_autoencoder.pth"

# --- MAIN SCRIPT ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare data
    df, scaler = data_handler.load_data(DATA_PATH)
    train_loader, test_loader = data_handler.create_dataloaders(df, SEQUENCE_LENGTH, BATCH_SIZE)

    # 2. Initialize Model
    model = model_arch.Autoencoder(SEQUENCE_LENGTH, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # 3. Train Model
    print("\n--- Starting Model Training ---")
    for epoch in range(EPOCHS):
        train_loss = engine.train_engine(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {train_loss:.4f}")
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

    # 4. Find Anomaly Threshold on training data
    threshold = utils.find_threshold(model, train_loader, device)

    # 5. Evaluate on Test Set
    print("\n--- Evaluating on Test Set ---")
    test_errors, true_labels = engine.evaluate_and_get_errors(model, test_loader, loss_fn, device)
    
    # 6. Get Predictions and Analyze
    predictions = (test_errors > threshold).astype(int)
    
    print("\n--- Classification Report ---")
    print(utils.classification_report(true_labels, predictions, target_names=['Normal', 'Anomaly']))
    
    # 7. Visualize Results
    utils.plot_reconstruction_errors(test_errors, threshold)
    utils.plot_confusion_matrix(true_labels, predictions)