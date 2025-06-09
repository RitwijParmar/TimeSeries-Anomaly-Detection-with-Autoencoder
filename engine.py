import torch

def train_engine(model, dataloader, optimizer, loss_fn, device):
    """
    Performs one epoch of training for the autoencoder.
    """
    model.train()
    total_loss = 0
    for seq, _ in dataloader:
        seq = seq.to(device)
        
        # Forward pass
        reconstructed = model(seq)
        loss = loss_fn(reconstructed, seq)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_and_get_errors(model, dataloader, loss_fn, device):
    """
    Evaluates the model and returns reconstruction errors.
    """
    model.eval()
    reconstruction_errors = []
    original_labels = []
    
    with torch.no_grad():
        for seq, labels in dataloader:
            seq = seq.to(device)
            reconstructed = model(seq)
            
            # Calculate loss for each item in the batch
            errors = torch.mean((reconstructed - seq) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
            original_labels.extend(labels.cpu().numpy())
            
    return np.array(reconstruction_errors), np.array(original_labels)