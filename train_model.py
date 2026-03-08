import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from model.model import SolarFlarePredictModel

model = SolarFlarePredictModel(input_features=25, seq_length=24).to(device)



#Loss Function
criterion_class = nn.CrossEntropyLoss()
criterion_time = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training Parameters
epochs = 50
reg_weight = 0.1    #Adjust this so regression loss doesn't dominate

for epoch in rang(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, target_class, target_time) in enumerate(train_loader):
        inputs = inputs.to(device)
        target_class = target_class.to(device).long()
        target_time = target_time.to(device).float()

        optimizer.zero_grad()

        #Forward
        pred_time, pred_class = model(inputs)


        #Classification Loss
        loss_class = criterion_class(pred_class, target_class)

        #Regression Loss (Only for actual flares)
        # We mask 'Quiet' regions (class 0) so they don't mess up the timer logic
        mask = (target_class > 0).float()
        loss_reg = (criterion_time(pred_time.squeeze(), target_time) * mask).mean()

        total_loss = loss_class + (reg_weight * loss_reg)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch+1/epochs}] Loss : {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "SolarFlarePredictModel.pt")