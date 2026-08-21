import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DATASET_FILE = 'dataset_sodium_normalized.npz' #dataset_lithium_normalized.npz
MODEL_SAVE_PATH = 'fno_sodium_model.pth'   #fno_lithium_model.pth
TRAIN_RESOLUTIONS = [(64, 100), (48, 75), (96, 150)]  # 64x100 'base'

# ==========================================
# 1. DATASET & DATALOADER
# ==========================================
class BatteryDataset(Dataset):
    """
    Custom PyTorch Dataset for PyBaMM normalized tensor.
    Input X: [Current, D_anode, D_cathode, D_electrolyte] (Shape: N x 4)
    Output Y: [Electrolyte Concentration (Space x Time)] (Shape: N x 64 x 100)
    """
    def __init__(self, npz_path):
        print(f"Loading dataset from '{npz_path}'...")
        dataset = np.load(npz_path)
        
        self.scaling_factors = dataset['scaling_factors'] 
        self.X = torch.tensor(dataset['X_norm'], dtype=torch.float32)
        # Add channel dimension to Y -> (N, 1, 64, 100)
        self.Y = torch.tensor(dataset['Y_norm'], dtype=torch.float32).unsqueeze(1) 
        
        print(f"Dataset loaded. X: {self.X.shape}, Y: {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(npz_path, batch_size=32):   # TO CHECK HERE
    full_dataset = BatteryDataset(npz_path)
    raw = np.load(npz_path)
    train_idx, val_idx, test_idx = raw['train_idx'], raw['val_idx'], raw['test_idx']

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.scaling_factors
# (Il notebook va sistemato: get_test_split() dovrebbe leggere test_idx dal file invece di richiamare random_split con seed 42)


# ==========================================
# 2. FOURIER NEURAL OPERATOR (FNO) MODEL
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=32):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.p = nn.Linear(6, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.mlp0 = nn.Conv2d(self.width, self.width, 1)
        self.mlp1 = nn.Conv2d(self.width, self.width, 1)
        self.mlp2 = nn.Conv2d(self.width, self.width, 1)
        self.mlp3 = nn.Conv2d(self.width, self.width, 1)
        
        self.q = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        
        gridx = gridx.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1) 
        gridy = gridy.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1) 
        return torch.cat([gridx, gridy], dim=1) 

    def forward(self, x_params, size_x=64, size_y=100):
        batchsize = x_params.shape[0]

        x = x_params.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size_x, size_y)
        grid = self.get_grid((batchsize, size_x, size_y), x_params.device)

        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.mlp0(x)
        x = F.gelu(x1 + x2) 

        x1 = self.conv1(x)
        x2 = self.mlp1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.mlp2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.mlp3(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 3, 1) 
        x = self.q(x)
        x = F.gelu(x)
        x = self.q2(x) 
        
        x = x.permute(0, 3, 1, 2) 
        return x


# ==========================================
# 3. LOSS FUNCTION & TRAINING LOOP
# ==========================================
class BoundedOutputLoss(nn.Module):
    """
    Custom loss combining standard data fitting (MSE) with a penalty
    that discourages predictions from dropping below the physically valid
    lower bound of the normalized concentration range (-1, corresponding
    to zero physical concentration). "Non-negativity constraint".
    """
    def __init__(self, mse_weight=1.0, bound_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.bound_weight = bound_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        data_loss = self.mse(predictions, targets)

        # Penalize predictions that drop below -1 (physically impossible concentration)
        negative_penalty = torch.relu(-(predictions + 1.0))
        bound_loss = torch.mean(negative_penalty ** 2)

        total_loss = (self.mse_weight * data_loss) + (self.bound_weight * bound_loss)
        return total_loss, data_loss, bound_loss


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training initialized. Using device: {device}")
    
    # Generate dataloaders directly from the function defined above
    train_loader, val_loader, test_loader, scaling_factors = get_dataloaders(DATASET_FILE, batch_size=BATCH_SIZE)
    
    print("Instantiating the Fourier Neural Operator...")
    model = FNO2d(modes1=16, modes2=16, width=32).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)    # HERE
    criterion = BoundedOutputLoss(mse_weight=1.0, bound_weight=0.5)

    run_name = MODEL_SAVE_PATH.replace('.pth', '')  # es. 'fno_sodium_model'
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    
    best_val_loss = float('inf')
    
    print("\nStarting Training Loop...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train() 
        train_loss_total = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() 

            # predictions = model(inputs)
            # loss, _, _ = criterion(predictions, targets)
            size_x, size_y = random.choice(TRAIN_RESOLUTIONS)
            targets_r = (targets if (size_x, size_y) == (64, 100)
                        else F.interpolate(targets, size=(size_x, size_y), mode='bicubic', align_corners=False))
            predictions = model(inputs, size_x=size_x, size_y=size_y)
            loss, *_ = criterion(predictions, targets_r)    # loss, _, _, _ = criterion(...)

            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            
        avg_train_loss = train_loss_total / len(train_loader)
        
        model.eval() 
        val_loss_total = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss, _, _ = criterion(predictions, targets)
                val_loss_total += loss.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'scaling_factors': scaling_factors
            }, MODEL_SAVE_PATH)
            saved_msg = " --> Checkpoint Saved!"
        else:
            saved_msg = ""
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{saved_msg}")
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved successfully to '{MODEL_SAVE_PATH}'.")
    print("To view training graphs, open a terminal and run: tensorboard --logdir=runs")

if __name__ == "__main__":
    train_model()