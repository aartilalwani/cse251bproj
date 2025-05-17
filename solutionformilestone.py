#!/usr/bin/env python
# coding: utf-8

# # Use This if you are using Kaggle Notebook

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load data

# In[2]:


# download the dataset to your folder or use it on kaggle notebook directly

train_file = np.load('./train.npz')
# train_file = np.load('/kaggle/input/cse-251-b-2025/train.npz')

train_data = train_file['data']
print("train_data's shape", train_data.shape)
test_file = np.load('./test_input.npz')
# test_file = np.load('/kaggle/input/cse-251-b-2025/test_input.npz')

test_data = test_file['data']
print("test_data's shape", test_data.shape)


# In[3]:


# plot one
import matplotlib.pyplot as plt

data_matrix = train_data[0]

for i in range(data_matrix.shape[0]):
    xs = data_matrix[i, :, 0]
    ys = data_matrix[i, :, 1]
    # trim all zeros
    xs = xs[xs != 0]
    ys = ys[ys != 0]
    # plot each line going from transparent to full
    plt.plot(xs, ys)

plt.show()


# # Prepare submission

# In[ ]:





# In[7]:


# say you have a model trained. we write a dummy model just to show useage

def dummy_model(input_data):
   return np.ones((2100, 60, 2))


output = dummy_model(test_data)
output.shape


# In[8]:


# reshape to fit desired format: (2100, 60, 2) -> (12600, 2)
dummy_output = output.reshape(-1, 2)
output_df = pd.DataFrame(dummy_output, columns=['x', 'y'])

# adding a necessary step to match index of your prediction to that of the solution key

output_df.index.name = 'index'

output_df.to_csv('dummy_submission.csv')


#  # Now you can submit to the leaderboard!

# ## Baseline 1: Constant Velocity 

# In[9]:


# Split x and y for train data.

train_x, train_y = train_data[..., :50, :], train_data[:, 0, 50:, :2]

# get the average velocity of the prediction agent
velocity_diff = train_x[...,1:, :2] - train_x[...,:-1, :2]
print(velocity_diff.shape)

constant_vel = np.mean(velocity_diff[:,0, :, :], axis=-2)
print(constant_vel.shape) 


# In[10]:


# create pred_y

pred_y = np.zeros((10000, 60, 2))
starting_point = train_x[:, 0, -1, :2] # shape (10000, 2)

for t in range(60):
    pred_y[:,t,:] = starting_point + (t+1) * constant_vel


# In[11]:


# calculate train loss

mse = ((train_y - pred_y)**2).mean()
print(mse)


# In[12]:


# prepare submission


# get the average velocity of the prediction agent
velocity_diff = test_data[...,1:, :2] - test_data[...,:-1, :2]
print(velocity_diff.shape)

constant_vel = np.mean(velocity_diff[:,0, :, :], axis=-2)
print(constant_vel.shape) 


# In[13]:


# create pred_y for test set

pred_y = np.zeros((2100, 60, 2))
starting_point = test_data[:, 0, -1, :2]

for t in range(60):
    pred_y[:,t,:] = starting_point + (t+1) * constant_vel


# In[14]:


# reshape to fit desired format: (2100, 60, 2) -> (12600, 2)
pred_output = pred_y.reshape(-1, 2)
output_df = pd.DataFrame(pred_output, columns=['x', 'y'])

# adding a necessary step to match index of your prediction to that of the solution key

output_df.index.name = 'index'

output_df.to_csv('constant_vel_submission.csv')


# ## Baseline 2: MLP without Normalization

# In[15]:


# Split x and y for train data.

train_x, train_y = train_data[..., :50, :], train_data[:, 0, 50:, :2]

print(train_x.shape, train_y.shape)


# In[37]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_features, output_features):
        super(MLP, self).__init__()
        
        # Define the layers
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, output_features)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.mlp(x)
        return x


# In[38]:


# Calculate the total number of features after flattening
input_features = 50 * 50 * 6  # = 5000
output_features = 60 * 2


# Create the model
model = MLP(input_features, output_features)

# Define loss function and optimizer
criterion = nn.MSELoss()  # For regression task

optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[18]:



# Example of how to prepare data and train the model

def train_model(model, x_train, y_train, batch_size=64, epochs=10):
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(x_train).reshape((-1, input_features))
    y_train_tensor = torch.FloatTensor(y_train).reshape((-1, output_features))
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    return model


# In[17]:


model = train_model(model, train_x, train_y)


# In[40]:



def predict(X_test):
    """Make predictions with the trained model"""
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).reshape((-1, input_features))
        predictions = model(X_test_tensor).reshape((-1, 60, 2))
    return predictions.numpy()

# Save model
def save_model(path="mlp_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model
def load_model(path="mlp_model.pth"):
    loaded_model = MLP()
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()
    return loaded_model


# In[41]:


pred_y = predict(test_data)

pred_output = pred_y.reshape(-1, 2)
output_df = pd.DataFrame(pred_output, columns=['x', 'y'])

# adding a necessary step to match index of your prediction to that of the solution key

output_df.index.name = 'index'

output_df.to_csv('mlp_baseline.csv')


# In[16]:





# In[42]:


# # Convert to tensors
# X_train_t = torch.tensor(train_x, dtype=torch.float32)

# # Create dataset (only input; target will be extracted per batch)
# dataset = TensorDataset(X_train_t)
# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# Make sure both are float32
train_x = train_x.astype('float32')
train_y = train_y.astype('float32')

# Convert to PyTorch tensors
x_tensor = torch.from_numpy(train_x)  # [10000, 50, 50, 6]
y_tensor = torch.from_numpy(train_y)  # [10000, 60, 2]

# Wrap both in a dataset
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# In[43]:


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# In[44]:


print(train_x.shape)
print(train_y.shape)


# In[ ]:





# In[45]:


class EgoMLP(nn.Module):
    def __init__(self, mean, std, input_dim=5, embed_dim=4, hidden_dim=128, output_len=60):
        super().__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.embed = nn.Embedding(10, embed_dim)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_dim + embed_dim) * 50, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len * 2)
        )

    def forward(self, x):
        ego = x[:, 0, :, :]
        cont = ego[:, :, :5]
        type_id = ego[:, :, 5].long().clamp(0, 9)
        cont_norm = (cont - self.mean.to(cont.device)) / self.std.to(cont.device)
        type_embed = self.embed(type_id)
        x_cat = torch.cat([cont_norm, type_embed], dim=-1)
        return self.mlp(x_cat).view(x.size(0), 60, 2)


# In[ ]:





# In[46]:


mean = torch.tensor(train_x[..., :5].reshape(-1, 5).mean(axis=0), dtype=torch.float32)
std = torch.tensor(train_x[..., :5].reshape(-1, 5).std(axis=0), dtype=torch.float32) + 1e-6

model = EgoMLP(mean, std).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss:.4f}")
save_model(path="egomlp.pth")


# In[47]:


def predict(X_test):
    """Make predictions with the trained model"""
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)#.reshape((-1, input_features))
        predictions = model(X_test_tensor)#.reshape((-1, 60, 2))
    return predictions
pred_y=predict(train_x)
pred_y=pred_y.cpu().numpy()
mse = ((train_y - pred_y)**2).mean()
print(mse)


# In[49]:


def predict(X_test):
    """Make predictions with the trained model"""
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)#.reshape((-1, input_features))
        predictions = model(X_test_tensor).reshape((-1, 60, 2))
    return predictions

pred_y = predict(test_data).cpu().numpy()

pred_output = pred_y.reshape(-1, 2)
output_df = pd.DataFrame(pred_output, columns=['x', 'y'])

# adding a necessary step to match index of your prediction to that of the solution key

output_df.index.name = 'index'

output_df.to_csv('egomlp.csv')


# In[52]:


class EgoLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2, output_len=60):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.output_len = output_len

    def forward(self, x, target=None):
        B = x.size(0)
        device = x.device
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1:, :2] #last pos

        outputs = []

        for _ in range(self.output_len):
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.out_proj(dec_out)
            outputs.append(pred)
            decoder_input = pred.detach()  # feeds back last prediction

        return torch.cat(outputs, dim=1)
    
model = EgoLSTM(input_dim=6, hidden_dim=128, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
model.to(device)

def train_lstm(model, loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch, target=y_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader.dataset):.6f}")
        
x_ego = train_x[:, 0, :, :]
x_tensor = torch.tensor(x_ego, dtype=torch.float32)
y_tensor = torch.tensor(train_y, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EgoLSTM(input_dim=6, hidden_dim=128, output_dim=2, output_len=60)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()


train_lstm(
    model=model,
    loader=loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=10,
)

model.eval()
all_preds = []
all_truths = []

with torch.no_grad():
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        all_preds.append(y_pred.cpu())
        all_truths.append(y_batch.cpu())

y_pred_all = torch.cat(all_preds, dim=0)
y_true_all = torch.cat(all_truths, dim=0)

criterion = torch.nn.MSELoss()
mse = criterion(y_pred_all, y_true_all).item()
print(f"Full-dataset MSE: {mse:.6f}")


# In[53]:


#for MLP
train_loss = [
    2221588407.5000, 439175944.6250, 113156211.2031, 13699838.4583, 2965594.6780,
    1361992.4091, 671744.0113, 376978.1496, 241718.5600, 165207.5567,
    118067.9013, 88328.6562, 68611.9059, 57559.4512, 45685.0950,
    38601.9883, 32871.6814, 27722.2265, 23460.2856, 19755.1929,
    16631.3623, 14162.3686, 11896.4059, 11186.9088, 10177.7540,
    9508.8496, 9717.7748, 9528.7479, 9664.5302, 9308.3598,
    9925.2469, 9141.5497, 9144.5098, 8808.2226, 9555.1781,
    10192.1273, 9358.4218, 8943.1566, 8417.0070, 9410.3190,
    8241.1752, 8385.0209, 8609.9663, 9003.4972, 9299.2672,
    7421.3288, 9436.8794, 7504.4267, 7627.0790, 8039.3199
]

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




