import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from pathlib import Path
import scipy.io
from scipy import signal
import mne
from mne.preprocessing import ICA

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

REAL_CHANNEL_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

def apply_filters(data, fs=512, lowcut=0.1, highcut=20.0, notch_freq=60.0):
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=0)
    b, a = signal.iirnotch(notch_freq / nyquist, Q=30.0)
    return signal.filtfilt(b, a, filtered, axis=0)

def apply_ica(data, ch_names, fs=512):
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)
    ica = ICA(n_components=20, random_state=seed, max_iter='auto')
    ica.fit(raw)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
    muscle_idx, _ = ica.find_bads_muscle(raw, threshold=1.0)
    ica.exclude = list(set(eog_idx + muscle_idx))
    cleaned = ica.apply(raw.copy()).get_data().T
    return cleaned

def apply_car(data):
    return data - np.mean(data, axis=1, keepdims=True)

def load_mat_file(filepath, fs=512, pre_ms=200, post_ms=800):
    mat = scipy.io.loadmat(filepath)['run'][0, 0]
    eeg = mat['eeg'][0, 0]
    events = mat['header'][0, 0]['EVENT'][0, 0]
    positions = events['POS'][0, 0].flatten()
    types = events['TYP'][0, 0].flatten()
    eeg = apply_filters(eeg, fs)
    eeg = apply_ica(eeg, REAL_CHANNEL_NAMES, fs)
    eeg = apply_car(eeg)
    pre, post = int(pre_ms * fs / 1000), int(post_ms * fs / 1000)
    correct, error = [], []
    correct_positions = []
    errp_positions = []
    for pos, typ in zip(positions, types):
        if pos - pre >= 0 and pos + post < len(eeg):
            epoch = eeg[pos - pre:pos + post]
            baseline = np.median(epoch[:pre], axis=0)
            epoch = epoch - baseline
            if typ in [5, 10]:
                correct.append(epoch)
                correct_positions.append(pos)
            elif typ in [6, 9]:
                error.append(epoch)
                errp_positions.append(pos)
    print(f"ErrP event positions in {os.path.basename(filepath)}: {errp_positions}")
    print(f"Correct event positions in {os.path.basename(filepath)}: {correct_positions}")
    return np.array(correct), np.array(error)

# Load all files
file_list = [
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject01_s2 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject01_s1 (2).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject02_s2 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject02_s1 (2).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject04_s2 (2).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject04_s1 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject05_s2 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject05_s1 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject06_s1 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject06_s2 (1).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject03_s2 (2).mat",
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\courses\Optimization\Project\dataset\Subject03_s1 (2).mat"
]
correct_epochs, error_epochs = [], []
for file in file_list:
    if Path(file).exists():
        c, e = load_mat_file(file)
        correct_epochs.extend(c)
        error_epochs.extend(e)

X = np.concatenate([correct_epochs, error_epochs])
y = np.array([0]*len(correct_epochs) + [1]*len(error_epochs))

scaler = StandardScaler()
for i in range(X.shape[2]):
    X[:, :, i] = scaler.fit_transform(X[:, :, i])
X_res, y_res = SMOTE(random_state=seed).fit_resample(X.reshape(len(X), -1), y)
X = X_res.reshape(-1, X.shape[1], X.shape[2])
y = y_res

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.75 * len(dataset))
train_ds, test_ds = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

print("EEG tensor shape (batch_size, channels, time, EEG_channels):", X_tensor.shape)
print("Number of samples:", X_tensor.shape[0])
print("Time steps (per trial):", X_tensor.shape[2])
print("EEG channels:", X_tensor.shape[3])
print("Number of batches in training set:", len(train_loader))
print("Number of batches in test set:", len(test_loader))

# Visualize EEG input as image
plt.figure(figsize=(10, 6))
plt.imshow(X[0].T, aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label='Amplitude (µV)')
plt.xlabel('Time (samples)')
plt.ylabel('EEG Channels')
plt.title('EEG Input Image (First Sample)')
plt.tight_layout()
plt.show()

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 1), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 1))
        self.drop1 = nn.Dropout(0.3)
        self.dw = nn.Conv2d(16, 16, (3, 1), groups=16, padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(16)
        self.pw = nn.Conv2d(16, 32, (3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 1))
        self.drop2 = nn.Dropout(0.3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(32 * (X.shape[1] // 4) * X.shape[2], 64)
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.drop1(self.pool1(self.bn1(torch.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.bn3(torch.relu(self.pw(self.bn2(torch.relu(self.dw(x))))))))
        x = self.flat(x)
        x = self.drop3(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_acc = 0
patience, trigger = 10, 0
train_acc, val_acc = [], []
y_true, y_pred = [], []

for epoch in range(50):
    model.train()
    correct = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == yb).sum().item()
    acc = correct / len(train_ds)
    train_acc.append(acc)

    model.eval()
    correct = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
            y_true.extend(yb.tolist())
            y_pred.extend(pred.argmax(1).tolist())
    val = correct / len(test_ds)
    val_acc.append(val)
    print(f"Epoch {epoch+1}: Train Acc={acc:.3f}, Val Acc={val:.3f}")
    if val > best_acc:
        best_acc = val
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print("Early stopping.")
            break

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Correct", "Error"]).plot()
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot ERP

def plot_average_erp(correct_epochs, error_epochs, channel_names, pre_samples, fs=512):
    time_axis = np.linspace(-200, 800, correct_epochs.shape[1] if len(correct_epochs) > 0 else error_epochs.shape[1])
    if len(correct_epochs) > 0:
        print(f"Averaging {len(correct_epochs)} Correct epochs")
        avg_correct = np.mean(correct_epochs, axis=0)
        plt.figure(figsize=(12, 4))
        for i in range(avg_correct.shape[1]):
            plt.plot(time_axis, avg_correct[:, i], label=channel_names[i], linewidth=1)
        plt.axvline(x=0, color='red', linestyle='--', label='Event')
        plt.title("Average Correct ERP")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if len(error_epochs) > 0:
        print(f"Averaging {len(error_epochs)} Error epochs")
        avg_error = np.mean(error_epochs, axis=0)
        plt.figure(figsize=(12, 4))
        for i in range(avg_error.shape[1]):
            plt.plot(time_axis, avg_error[:, i], label=channel_names[i], linewidth=1)
        plt.axvline(x=0, color='red', linestyle='--', label='Event')
        plt.title("Average Error ERP")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

pre_samples = int(200 * 512 / 1000)
plot_average_erp(np.array(correct_epochs), np.array(error_epochs), REAL_CHANNEL_NAMES, pre_samples)

# Plot single epochs for visual reference like image example
if len(correct_epochs) > 0:
    plt.figure(figsize=(12, 4))
    for i in range(len(REAL_CHANNEL_NAMES)):
        plt.plot(np.linspace(-200, 800, correct_epochs[0].shape[0]), correct_epochs[0][:, i], label=REAL_CHANNEL_NAMES[i])
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f"Correct Epoch at Position {pre_samples}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()

if len(error_epochs) > 0:
    plt.figure(figsize=(12, 4))
    for i in range(len(REAL_CHANNEL_NAMES)):
        plt.plot(np.linspace(-200, 800, error_epochs[0].shape[0]), error_epochs[0][:, i], label=REAL_CHANNEL_NAMES[i])
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f"Error Epoch at Position {pre_samples}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()
    print(f"Image size per trial: Time Steps = {X.shape[1]}, Channels = {X.shape[2]}")

