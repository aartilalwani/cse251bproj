#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sources: Used Generative AI to help write some modeling code

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# download the dataset to your folder or use it on kaggle notebook directly

train_file = np.load('/kaggle/input/251b-data/train.npz')

train_data = train_file['data']
print("train_data's shape", train_data.shape)
test_file = np.load('/kaggle/input/251b-data/test_input.npz')

test_data = test_file['data']
print("test_data's shape", test_data.shape)


# In[7]:


#!pip install torch_geometric


# In[8]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return out, h[-1]

class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        query = self.query.unsqueeze(0).expand(lstm_out.size(0), -1).unsqueeze(-1) 
        scores = torch.bmm(lstm_out, query).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) 
        return (weights * lstm_out).sum(dim=1) 
def make_mlp(hidden_size, output_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim)
    )

class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size, future_len, output_size):
        super().__init__()
        self.future_len = future_len
        self.output_size = output_size
        self.pos_head = make_mlp(hidden_size, future_len * output_size)
        self.vel_head = make_mlp(hidden_size, future_len * output_size)

    def forward(self, h):
        pos = self.pos_head(h)
        vel = self.vel_head(h)
        return pos.view(-1, self.future_len, self.output_size), vel.view(-1, self.future_len, self.output_size)

class AblationLSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, future_len=60, output_size=2,
                 use_attention=False, use_multitask=False, use_residual=True, num_layers=1):
        super().__init__()
        self.future_len = future_len
        self.output_size = output_size
        self.use_attention = use_attention
        self.use_multitask = use_multitask
        self.use_residual = use_residual

        self.encoder = BaseLSTMEncoder(input_size, hidden_size, num_layers)

        if use_attention:
            self.attn = DotProductAttention(hidden_size)

        if use_multitask:
            self.head = MultiTaskHead(hidden_size, future_len, output_size)
        else:
            self.head = make_mlp(hidden_size, future_len * output_size)

    def forward(self, x):
        B = x.size(0)
        last_pos = x[:, -1, :2]

        lstm_out, h_final = self.encoder(x)
        h = self.attn(lstm_out) if self.use_attention else h_final

        if self.use_multitask:
            delta_pos, delta_vel = self.head(h)
        else:
            delta = self.head(h).view(B, self.future_len, self.output_size)
            delta_pos, delta_vel = delta, None

        if self.use_residual:
            pred_pos = last_pos.unsqueeze(1) + torch.cumsum(delta_pos, dim=1)
        else:
            pred_pos = delta_pos

        return pred_pos, delta_vel


# In[9]:




def train_ablation_model_with_val(model, train_loader, val_loader, epochs=300, lr=1e-4, alpha=0.1,
                                  device='cuda', early_stop_patience=6, save_path=None):
    import torch
    import torch.nn as nn

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pos_loss = 0
        train_vel_loss = 0

        all_train_pred_pos = []
        all_train_gt_pos = []
        all_train_pred_vel = []
        all_train_gt_vel = []
        all_train_origin = []
        all_train_scale = []

        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            origin = batch["origin"].to(device)
            scale = batch["scale"].to(device)

            yb_pos = yb[:, :, :2]
            yb_vel = yb[:, :, 2:]

            pred_pos, pred_vel = model(xb)
            pos_loss = criterion(pred_pos, yb_pos)
            vel_loss = criterion(pred_vel, yb_vel) if model.use_multitask and pred_vel is not None else 0.0
            loss = pos_loss + alpha * vel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pos_loss += pos_loss.item()
            train_vel_loss += vel_loss if isinstance(vel_loss, float) else vel_loss.item()

            all_train_pred_pos.append(pred_pos)
            all_train_gt_pos.append(yb_pos)
            all_train_origin.append(origin)
            all_train_scale.append(scale)
            if model.use_multitask and pred_vel is not None:
                all_train_pred_vel.append(pred_vel)
                all_train_gt_vel.append(yb_vel)

        model.eval()
        val_loss = 0
        val_pos_loss = 0
        val_vel_loss = 0

        all_pred_pos = []
        all_gt_pos = []
        all_pred_vel = []
        all_gt_vel = []
        all_origin = []
        all_scale = []

        with torch.no_grad():
            for batch in val_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                origin = batch["origin"].to(device)
                scale = batch["scale"].to(device)

                yb_pos = yb[:, :, :2]
                yb_vel = yb[:, :, 2:]

                pred_pos, pred_vel = model(xb)
                pos_loss = criterion(pred_pos, yb_pos)
                vel_loss = criterion(pred_vel, yb_vel) if model.use_multitask and pred_vel is not None else 0.0

                val_loss += (pos_loss + alpha * vel_loss).item()
                val_pos_loss += pos_loss.item()
                val_vel_loss += vel_loss if isinstance(vel_loss, float) else vel_loss.item()

                all_pred_pos.append(pred_pos)
                all_gt_pos.append(yb_pos)
                all_origin.append(origin)
                all_scale.append(scale)
                if model.use_multitask and pred_vel is not None:
                    all_pred_vel.append(pred_vel)
                    all_gt_vel.append(yb_vel)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_pos = train_pos_loss / len(train_loader)
        avg_train_vel = train_vel_loss / len(train_loader)
        avg_val_pos = val_pos_loss / len(val_loader)
        avg_val_vel = val_vel_loss / len(val_loader)

        # Logging
        if epoch == 0 or epoch % 20 == 0 or epoch == epochs - 1 or avg_val_pos < best_val_loss:
            # --- Denormalize Train ---
            pred_pos = torch.cat(all_train_pred_pos, dim=0)
            gt_pos = torch.cat(all_train_gt_pos, dim=0)
            origin = torch.cat(all_train_origin, dim=0).unsqueeze(1)  # (B, 1, 2)
            scale = torch.cat(all_train_scale, dim=0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

            pred_pos_denorm = pred_pos * scale + origin
            gt_pos_denorm = gt_pos * scale + origin
            train_pos_mse_denorm = ((pred_pos_denorm - gt_pos_denorm) ** 2).mean().item()

            if model.use_multitask and all_train_pred_vel:
                pred_vel = torch.cat(all_train_pred_vel, dim=0)
                gt_vel = torch.cat(all_train_gt_vel, dim=0)
                pred_vel_denorm = pred_vel * scale
                gt_vel_denorm = gt_vel * scale
                train_vel_mse_denorm = ((pred_vel_denorm - gt_vel_denorm) ** 2).mean().item()
            else:
                train_vel_mse_denorm = 0.0

            # --- Denormalize Val ---
            pred_pos = torch.cat(all_pred_pos, dim=0)
            gt_pos = torch.cat(all_gt_pos, dim=0)
            origin = torch.cat(all_origin, dim=0).unsqueeze(1)  # (B, 1, 2)
            scale = torch.cat(all_scale, dim=0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

            pred_pos_denorm = pred_pos * scale + origin
            gt_pos_denorm = gt_pos * scale + origin
            val_pos_mse_denorm = ((pred_pos_denorm - gt_pos_denorm) ** 2).mean().item()

            if model.use_multitask and all_pred_vel:
                pred_vel = torch.cat(all_pred_vel, dim=0)
                gt_vel = torch.cat(all_gt_vel, dim=0)
                pred_vel_denorm = pred_vel * scale
                gt_vel_denorm = gt_vel * scale
                val_vel_mse_denorm = ((pred_vel_denorm - gt_vel_denorm) ** 2).mean().item()
            else:
                val_vel_mse_denorm = 0.0

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {avg_train_loss:.6f} | Train Pos: {avg_train_pos:.6f} | Train Vel: {avg_train_vel:.6f} | "
                f"Val Pos: {avg_val_pos:.6f} | Val Vel: {avg_val_vel:.6f} || "
                f"[Denorm MSE] Train Pos (m²): {train_pos_mse_denorm:.4f} | Train Vel (m²/s²): {train_vel_mse_denorm:.4f} | "
                f"Val Pos (m²): {val_pos_mse_denorm:.4f} | Val Vel (m²/s²): {val_vel_mse_denorm:.4f}"
            )

        # Early stopping
        if avg_val_pos < best_val_loss:
            best_val_loss = avg_val_pos
            best_model_state = model.state_dict()
            patience_counter = 0
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"✅ Model saved at epoch {epoch+1} to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


# In[10]:


from torch.utils.data import TensorDataset, DataLoader, random_split
def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        ego_hist = scene[0, :50, :].copy()   # shape (50, 6)
        future = scene[0, 50:, :4].copy()    # shape (60, 4) → x, y, vx, vy

        # Data augmentation (only for training)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)

                # Rotate pos, vel, future
                ego_hist[:, :2] = ego_hist[:, :2] @ R
                ego_hist[:, 2:4] = ego_hist[:, 2:4] @ R
                future[:, :2] = future[:, :2] @ R
                future[:, 2:4] = future[:, 2:4] @ R

                # Rotate heading
                ego_hist[:, 4] = wrap_angle(ego_hist[:, 4] + theta)

            if np.random.rand() < 0.5:
                ego_hist[:, 0] *= -1  # x pos
                ego_hist[:, 2] *= -1  # x vel
                future[:, 0] *= -1    # x pos
                future[:, 2] *= -1    # x vel

        # Convert heading to sin/cos
        heading = ego_hist[:, 4]
        sin_theta = np.sin(heading)
        cos_theta = np.cos(heading)

        # Drop vehicle type, replace heading with sin/cos
        ego_hist = np.concatenate([
            ego_hist[:, :4],                  # x, y, vx, vy
            sin_theta[:, None],              # sin(θ)
            cos_theta[:, None]               # cos(θ)
        ], axis=-1)  # → shape (50, 6)

        # Compute acceleration
        delta_t = 0.1
        vx = ego_hist[:, 2]
        vy = ego_hist[:, 3]
        ax = np.diff(vx, prepend=vx[0]) / delta_t
        ay = np.diff(vy, prepend=vy[0]) / delta_t

        ego_hist = np.concatenate([
            ego_hist,                        # x, y, vx, vy, sin(θ), cos(θ)
            ax[:, None], ay[:, None]        # ax, ay
        ], axis=-1)  # → shape (50, 8)

        # Normalize position/velocity/acceleration
        origin = ego_hist[49, :2].copy()
        ego_hist[:, :2] -= origin
        future[:, :2] -= origin

        ego_hist[:, [0, 1, 2, 3, 6, 7]] /= self.scale
        future /= self.scale

        return {
            "x": torch.tensor(ego_hist, dtype=torch.float32),     # (50, 8)
            "y": torch.tensor(future, dtype=torch.float32),       # (60, 4)
            "origin": torch.tensor(origin, dtype=torch.float32),
            "scale": torch.tensor(self.scale, dtype=torch.float32),
        }

scale = 10.0

N = len(train_data)
val_size = int(0.1 * N)
train_size = N - val_size

train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=True)
val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)





# In[11]:


baselinemodel = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=False,        
    use_multitask=False,       
    use_residual=False,
)

train_ablation_model_with_val(
    model=baselinemodel,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[12]:


residualmodel = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=False,        
    use_multitask=False,        
    use_residual=True,
)

train_ablation_model_with_val(
    model=residualmodel,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[13]:


attentionresidualmodel = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=True,       
    use_multitask=False,       
    use_residual=True,
)

train_ablation_model_with_val(
    model=attentionresidualmodel,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[14]:


dualheadattentionresidualmodel = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=True,        
    use_multitask=True,        
    use_residual=True,
)

train_ablation_model_with_val(
    model=dualheadattentionresidualmodel,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[15]:


dualheadresidualmodel = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=False,        
    use_multitask=True,        
    use_residual=True,
)

train_ablation_model_with_val(
    model=dualheadresidualmodel,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[ ]:


import time
def printtimetrain_ablation_model_with_val(model, train_loader, val_loader, epochs=1, lr=1e-4, alpha=0.1,
                                  device='cuda', early_stop_patience=6, save_path=None):
    import torch
    import torch.nn as nn

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pos_loss = 0
        train_vel_loss = 0

        all_train_pred_pos = []
        all_train_gt_pos = []
        all_train_pred_vel = []
        all_train_gt_vel = []
        all_train_origin = []
        all_train_scale = []

        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)
            origin = batch["origin"].to(device)
            scale = batch["scale"].to(device)

            yb_pos = yb[:, :, :2]
            yb_vel = yb[:, :, 2:]

            pred_pos, pred_vel = model(xb)
            pos_loss = criterion(pred_pos, yb_pos)
            vel_loss = criterion(pred_vel, yb_vel) if model.use_multitask and pred_vel is not None else 0.0
            loss = pos_loss + alpha * vel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pos_loss += pos_loss.item()
            train_vel_loss += vel_loss if isinstance(vel_loss, float) else vel_loss.item()

            all_train_pred_pos.append(pred_pos)
            all_train_gt_pos.append(yb_pos)
            all_train_origin.append(origin)
            all_train_scale.append(scale)
            if model.use_multitask and pred_vel is not None:
                all_train_pred_vel.append(pred_vel)
                all_train_gt_vel.append(yb_vel)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function took {elapsed_time:.8f} seconds")
        return
        model.eval()
        val_loss = 0
        val_pos_loss = 0
        val_vel_loss = 0

        all_pred_pos = []
        all_gt_pos = []
        all_pred_vel = []
        all_gt_vel = []
        all_origin = []
        all_scale = []

        with torch.no_grad():
            for batch in val_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                origin = batch["origin"].to(device)
                scale = batch["scale"].to(device)

                yb_pos = yb[:, :, :2]
                yb_vel = yb[:, :, 2:]

                pred_pos, pred_vel = model(xb)
                pos_loss = criterion(pred_pos, yb_pos)
                vel_loss = criterion(pred_vel, yb_vel) if model.use_multitask and pred_vel is not None else 0.0

                val_loss += (pos_loss + alpha * vel_loss).item()
                val_pos_loss += pos_loss.item()
                val_vel_loss += vel_loss if isinstance(vel_loss, float) else vel_loss.item()

                all_pred_pos.append(pred_pos)
                all_gt_pos.append(yb_pos)
                all_origin.append(origin)
                all_scale.append(scale)
                if model.use_multitask and pred_vel is not None:
                    all_pred_vel.append(pred_vel)
                    all_gt_vel.append(yb_vel)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_pos = train_pos_loss / len(train_loader)
        avg_train_vel = train_vel_loss / len(train_loader)
        avg_val_pos = val_pos_loss / len(val_loader)
        avg_val_vel = val_vel_loss / len(val_loader)

        # Logging
        if epoch == 0 or epoch % 20 == 0 or epoch == epochs - 1 or avg_val_pos < best_val_loss:
            # --- Denormalize Train ---
            pred_pos = torch.cat(all_train_pred_pos, dim=0)
            gt_pos = torch.cat(all_train_gt_pos, dim=0)
            origin = torch.cat(all_train_origin, dim=0).unsqueeze(1)  # (B, 1, 2)
            scale = torch.cat(all_train_scale, dim=0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

            pred_pos_denorm = pred_pos * scale + origin
            gt_pos_denorm = gt_pos * scale + origin
            train_pos_mse_denorm = ((pred_pos_denorm - gt_pos_denorm) ** 2).mean().item()

            if model.use_multitask and all_train_pred_vel:
                pred_vel = torch.cat(all_train_pred_vel, dim=0)
                gt_vel = torch.cat(all_train_gt_vel, dim=0)
                pred_vel_denorm = pred_vel * scale
                gt_vel_denorm = gt_vel * scale
                train_vel_mse_denorm = ((pred_vel_denorm - gt_vel_denorm) ** 2).mean().item()
            else:
                train_vel_mse_denorm = 0.0

            # --- Denormalize Val ---
            pred_pos = torch.cat(all_pred_pos, dim=0)
            gt_pos = torch.cat(all_gt_pos, dim=0)
            origin = torch.cat(all_origin, dim=0).unsqueeze(1)  # (B, 1, 2)
            scale = torch.cat(all_scale, dim=0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

            pred_pos_denorm = pred_pos * scale + origin
            gt_pos_denorm = gt_pos * scale + origin
            val_pos_mse_denorm = ((pred_pos_denorm - gt_pos_denorm) ** 2).mean().item()

            if model.use_multitask and all_pred_vel:
                pred_vel = torch.cat(all_pred_vel, dim=0)
                gt_vel = torch.cat(all_gt_vel, dim=0)
                pred_vel_denorm = pred_vel * scale
                gt_vel_denorm = gt_vel * scale
                val_vel_mse_denorm = ((pred_vel_denorm - gt_vel_denorm) ** 2).mean().item()
            else:
                val_vel_mse_denorm = 0.0

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {avg_train_loss:.6f} | Train Pos: {avg_train_pos:.6f} | Train Vel: {avg_train_vel:.6f} | "
                f"Val Pos: {avg_val_pos:.6f} | Val Vel: {avg_val_vel:.6f} || "
                f"[Denorm MSE] Train Pos (m²): {train_pos_mse_denorm:.4f} | Train Vel (m²/s²): {train_vel_mse_denorm:.4f} | "
                f"Val Pos (m²): {val_pos_mse_denorm:.4f} | Val Vel (m²/s²): {val_vel_mse_denorm:.4f}"
            )

        # Early stopping
        if avg_val_pos < best_val_loss:
            best_val_loss = avg_val_pos
            best_model_state = model.state_dict()
            patience_counter = 0
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"✅ Model saved at epoch {epoch+1} to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

dualheadresidualmodel2 = AblationLSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    future_len=60,
    output_size=2,
    use_attention=False,        
    use_multitask=True,        
    use_residual=True,
)

printtimetrain_ablation_model_with_val(
    model=dualheadresidualmodel2,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=1e-4,
    alpha=0.1,
    device='cuda',
    early_stop_patience=6,
    save_path=None
)


# In[ ]:





# In[30]:


import random
model1 = baselinemodel
model2 = residualmodel
model3 = attentionresidualmodel
model4 = dualheadresidualmodel
batch = next(iter(val_loader))  # one batch
batch_size = batch["x"].shape[0]

# Pick 4 unique random indices from the batch
rand_indices = random.sample(range(batch_size), 4)

# Stack selected samples into a new batch of 4
sample = {
    k: torch.stack([v[i] for i in rand_indices]).to(device)
    for k, v in batch.items()
}
import matplotlib.pyplot as plt

model1.eval()
model2.eval()
model3.eval()
model4.eval()

with torch.no_grad():
    pred_pos1, _ = model1(sample["x"])
    pred_pos2, _ = model2(sample["x"])
    pred_pos3, _ = model3(sample["x"])
    pred_pos4, _ = model4(sample["x"])

gt_pos = sample["y"][:, :, :2]                       # shape: (4, 60, 2)
origin = sample["origin"].unsqueeze(1)              # shape: (4, 1, 2)
scale = sample["scale"].unsqueeze(1).unsqueeze(2)   # shape: (4, 1, 1)

# Denormalize predictions
pred_pos1 = pred_pos1 * scale + origin
pred_pos2 = pred_pos2 * scale + origin
pred_pos3 = pred_pos3 * scale + origin
pred_pos4 = pred_pos4 * scale + origin
gt_pos = gt_pos * scale + origin

# Convert to numpy
pred1_np = pred_pos1.cpu().numpy()
pred2_np = pred_pos2.cpu().numpy()
pred3_np = pred_pos3.cpu().numpy()
pred4_np = pred_pos4.cpu().numpy()
gt_np = gt_pos.cpu().numpy()
colors = {
    'gt': 'black',
    'baseline': 'red',
    'resid': 'blue',
    'attnresid': 'green',
    'dualheadresid': 'orange'  # 'y' (yellow) can be hard to see
}

# Loop through each sample in the batch
for i in range(pred1_np.shape[0]):
    plt.figure(figsize=(8, 6))
    plt.plot(gt_np[i, :, 0], gt_np[i, :, 1], color=colors['gt'], linestyle='-', label='Ground Truth')
    plt.plot(pred1_np[i, :, 0], pred1_np[i, :, 1], color=colors['baseline'], linestyle='-', label='Baseline')
    plt.plot(pred2_np[i, :, 0], pred2_np[i, :, 1], color=colors['resid'], linestyle='-', label='Resid')
    plt.plot(pred3_np[i, :, 0], pred3_np[i, :, 1], color=colors['attnresid'], linestyle='-', label='AttnResid')
    plt.plot(pred4_np[i, :, 0], pred4_np[i, :, 1], color=colors['dualheadresid'], linestyle='-', label='DualHeadResid')

    plt.legend()
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Sample Trajectory Prediction")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# In[ ]:


# Sliding Window Model
class SlidingResidualEgoLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=1, 
                 future_len=60, step_size=5, output_size=2):
        super().__init__()
        self.future_len = future_len
        self.step_size = step_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, step_size * output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        last_pos = x[:, -1, :2]
        _, (h, c) = self.lstm(x)
        pred_traj = []
        current_pos = last_pos

        for _ in range(self.future_len // self.step_size):
            delta = self.fc(h[-1])
            delta = delta.view(batch_size, self.step_size, 2)
            current_pos = current_pos.unsqueeze(1) + torch.cumsum(delta, dim=1)
            pred_traj.append(current_pos)
            current_pos = current_pos[:, -1, :] 

        return torch.cat(pred_traj, dim=1)
# Weighted Tower Model
class WeightedTripleTowerLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, future_len=60, output_size=2):
        super().__init__()
        self.future_len = future_len
        self.hidden_size = hidden_size

        self.motion_lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.spatial_lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.unified_lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, batch_first=True)

        self.attn_weights = nn.Linear(hidden_size, 1)

        # Fully connected decoder
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, future_len * output_size)
        )

    def forward(self, x):
        motion = x[:, :, 2:6]            # vx, vy, ax, ay
        spatial = x[:, :, [0, 1, 4, 5]]  # x, y, sin(θ), cos(θ)

        _, (h_motion, _) = self.motion_lstm(motion)
        _, (h_spatial, _) = self.spatial_lstm(spatial)
        _, (h_unified, _) = self.unified_lstm(x)
        h_stack = torch.stack([h_motion[-1], h_spatial[-1], h_unified[-1]], dim=1)
        scores = self.attn_weights(h_stack)           # (B, 3, 1)
        weights = torch.softmax(scores, dim=1)        # (B, 3, 1)
        h_final = (weights * h_stack).sum(dim=1)
        delta = self.fc(h_final).view(-1, self.future_len, 2)
        last_pos = x[:, -1, :2]
        return last_pos.unsqueeze(1) + torch.cumsum(delta, dim=1)


# In[ ]:


import itertools
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset

class ResidualEgoLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=1, future_len=60, output_size=2, mlp_depth=2):
        super().__init__()
        self.future_len = future_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        layers = []
        for _ in range(mlp_depth - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, future_len * output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        last_pos = x[:, -1, :2]  # (B, 2)
        _, (h, _) = self.lstm(x)
        h_final = h[-1]  # (B, hidden_size)
        delta = self.fc(h_final).view(-1, self.future_len, 2)
        return last_pos.unsqueeze(1) + torch.cumsum(delta, dim=1)

def evaluate(model, val_loader, device='cuda'):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import itertools
import copy

def grid_search_with_early_stopping(x_tensor, y_tensor, device='cuda'):
    hidden_sizes = [96, 112, 128, 140, 256]
    num_layers_list = [1,2,3]
    lrs = [1e-2, 1e-3, 5e-4]
    patience = 6
    max_epochs = 100

    val_frac = 0.1
    N = len(x_tensor)
    val_size = int(N * val_frac)
    train_size = N - val_size
    train_dataset, val_dataset = random_split(TensorDataset(x_tensor, y_tensor), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    best_val_loss = float('inf')
    best_config = None
    best_model_state = None

    for hs, nl, lr in itertools.product(hidden_sizes, num_layers_list, lrs):
        print(f"\nTesting config: hidden_size={hs}, num_layers={nl}, lr={lr}")
        model = ResidualEgoLSTM(input_size=8, hidden_size=hs, num_layers=nl).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_epoch_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(max_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = evaluate(model, val_loader, device=device)
            print(f"  Epoch {epoch+1}, Val Loss: {val_loss:.6f}")

            if val_loss < best_epoch_loss:
                best_epoch_loss = val_loss
                no_improve_epochs = 0
                # Save model state for this config
                best_model_snapshot = copy.deepcopy(model.state_dict())
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print("  Early stopping.")
                    break

        if best_epoch_loss < best_val_loss:
            best_val_loss = best_epoch_loss
            best_config = (hs, nl, lr)
            best_model_state = best_model_snapshot

    print("\nBest configuration:")
    print(f"  Hidden size: {best_config[0]}, Num layers: {best_config[1]}, LR: {best_config[2]}")
    print(f"  Best Val Loss: {best_val_loss:.6f}")

    # Load best model state
    final_model = ResidualEgoLSTM(input_size=4, hidden_size=best_config[0], num_layers=best_config[1]).to(device)
    final_model.load_state_dict(best_model_state)

    return final_model, best_config

final_model, best_config = grid_search_with_early_stopping(x_tensor, y_tensor, device='cuda')

