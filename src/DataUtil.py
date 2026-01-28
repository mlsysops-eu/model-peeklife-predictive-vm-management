import numpy as np
import torch
from torch.utils.data import Dataset

class UtilizationDataset(Dataset):
    def __init__(self, data, input_length, forecast_length=6, stride=6,
                 normalize=True, max_lifetime_seconds=None):
        self.input_length = int(input_length)
        self.forecast_length = int(forecast_length)
        self.stride = int(stride)

        self.data = data.copy()

        # CPU - original CSV is 0-100, normalize to 0-1
        self.data["AvgCPU"] = self.data["AvgCPU"] / 100.0
        self.data["MaxCPU"] = self.data["MaxCPU"] / 100.0

        # MaxCPU_so_far norm
        if "MaxCPU_so_far" not in self.data.columns:
            self.data["MaxCPU_so_far"] = self.data["MaxCPU"]
        self.data["MaxCPU_so_far"] = self.data["MaxCPU_so_far"] / 100.0

        # lifetime
        if max_lifetime_seconds is None:
            self.max_lifetime = float(self.data["lifetime_seconds"].max())
        else:
            self.max_lifetime = float(max_lifetime_seconds)

        # Time ordering
        sort_col = "TimeStamp" if "TimeStamp" in self.data.columns else (
            "time_relative_seconds" if "time_relative_seconds" in self.data.columns else None
        )

        self.vm_data = {
            vm_id: (df.sort_values(by=sort_col) if sort_col else df)
            for vm_id, df in self.data.groupby("VMID")
        }

        self.valid_sequences = self._generate_valid_sequences()

    def _generate_valid_sequences(self):
        valid_sequences = []
        for vm_id, vm_data in self.vm_data.items():
            num_valid_windows = (len(vm_data) - self.input_length - self.forecast_length) // self.stride + 1
            for start_idx in range(0, num_valid_windows * self.stride, self.stride):
                valid_sequences.append((vm_id, start_idx))
        return valid_sequences

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        vm_id, start_idx = self.valid_sequences[idx]
        vm_data = self.vm_data[vm_id]

        historical = vm_data.iloc[start_idx:start_idx + self.input_length]
        future = vm_data.iloc[start_idx + self.input_length:start_idx + self.input_length + self.forecast_length]

        avg_cpu = historical["AvgCPU"].values
        max_cpu = historical["MaxCPU"].values
        max_cpu_so_far = float(historical["MaxCPU_so_far"].iloc[-1])

        future_avg = future["AvgCPU"].values
        future_max = future["MaxCPU"].values
        future_util = np.stack([future_avg, future_max], axis=-1).astype(np.float32)

        total_life_s = float(historical["lifetime_seconds"].iloc[-1])
        running_s    = float(historical["time_relative_seconds"].iloc[-1])
        remaining_s  = max(0.0, total_life_s - running_s)

        den = max(self.max_lifetime, 1e-6)
        lifetime_normalized = remaining_s / den
        aux_features = np.array([running_s / den, max_cpu_so_far], dtype=np.float32)

        return {
            "inputs": np.stack([avg_cpu, max_cpu], axis=-1).astype(np.float32),
            "future_utils": future_util,
            "lifetime": np.float32(lifetime_normalized),
            "aux_features": aux_features
        }

def custom_collate_fn(batch):
    inputs = torch.stack([torch.tensor(item["inputs"]) for item in batch])
    future_utils = torch.stack([torch.tensor(item["future_utils"]) for item in batch])
    lifetimes = torch.tensor([item["lifetime"] for item in batch])
    aux_features = torch.stack([torch.tensor(item["aux_features"]) for item in batch])
    return {"inputs": inputs, "future_utils": future_utils, "lifetime": lifetimes, "aux_features": aux_features}
