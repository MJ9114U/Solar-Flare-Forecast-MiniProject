import torch
import torch.nn as nn
import torch.nn.functional as F

class SolarFlarePredictModel(nn.Module):
    def __init__(self, input_features, seq_length):
        super(SolarFlarePredictModel, self).__init__()

        #CNN: Extracts local "spatial" patterns from magnetic features
        # We use padding=1 to keep the timeline consistent
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)

        #MaxPool: Reduces temporal resolution (summarizes the timeline)
        self.pool = nn.MaxPool1d(kernel_size=2)

        #LSTM: Temporal processing of the CNN-extracted features
        # The input_size MUST match the CNN out_channels (32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        #Dense Bridge
        self.fc_shared = nn.Linear(64, 32)

        #Multitask Outputs
        self.time_to_flare = nn.Linear(32, 1) #Regression Head
        self.flare_class = nn.Linear(32, 4) #Classification Head (4 classes: B, C, M, X)
        

    def forward(self, x):
        # x input: (Batch, Seq_Len, Features)
        
        # --- SPATIAL EXTRACTION (CNN) ---
        # CNN expects (Batch, Features, Seq_Len)
        x = x.transpose(1, 2) 
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Temporal resolution halved here

        # --- TEMPORAL PROCESSING (LSTM) ---
        # LSTM expects (Batch, Seq_Len, Features)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # Use only the last hidden state (summary of the whole window)
        x = lstm[:, -1, :]

        # --- OUTPUT HEADS ---
        x = F.relu(self.fc_shared(x))

        time_pred = self.time_to_flare(x)
        class_pred = self.flare_class(x)
        
        return time_pred, class_pred