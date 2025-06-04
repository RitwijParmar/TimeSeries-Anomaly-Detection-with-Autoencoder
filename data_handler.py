import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Loads and preprocesses the Numenta Anomaly Benchmark dataset."""
    df = pd.read_csv(filepath, parse_dates=True, index_col='timestamp')
    
    # Normalize the 'value' column
    scaler = MinMaxScaler()
    df['value_scaled'] = scaler.fit_transform(df[['value']])
    
    return df, scaler

def create_sequences(data, sequence_length):
    """Creates overlapping sequences from time-series data."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def create_dataloaders(df, sequence_length, batch_size):
    """Creates training and testing dataloaders."""
    data_scaled = df['value_scaled'].values
    
    # For anomaly detection, we train on the normal data
    normal_data_scaled = df[df['label'] == 0]['value_scaled'].values
    
    train_sequences = create_sequences(normal_data_scaled, sequence_length)
    test_sequences = create_sequences(data_scaled, sequence_length)
    
    # Create Tensors
    X_train = torch.tensor(train_sequences, dtype=torch.float32)
    X_test = torch.tensor(test_sequences, dtype=torch.float32)
    y_test = torch.tensor(df['label'].values[sequence_length:], dtype=torch.float32)
    
    # Create Datasets
    train_dataset = TensorDataset(X_train, X_train) # Autoencoder learns to reconstruct itself
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader