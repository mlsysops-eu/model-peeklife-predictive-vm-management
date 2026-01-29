# PeakLife (ONNX) — VM Utilization + Remaining Lifetime Predictor

This repository contains **PeakLife**, a lightweight neural model exported to **ONNX** for portable inference.  
Given the historic utilization information for a VM, PeakLife predicts:

- **Future CPU utilization**: **AvgCPU** and **MaxCPU** (normalized)
- **Remaining lifetime**: normalized remaining lifetime (and seconds via scaling)

The repo includes a minimal demo pipeline that loads a small CSV (`demodata.csv`), preprocesses it to the model’s expected inputs, runs ONNX inference, and prints the results.

## Project Structure

```text
.
├── model/
│   ├── peaklife.onnx            # ONNX model
│   └── model_config.json        # Model card & Configuration
├── src/ (Optional)              # Helpers for data preparation
│   ├── DataUtil.py        
│   └── prepare_demodata.py     
├── demo.py                      # Main entry point for inference demo
├── demodata.csv                 # Demo dataset
├── requirements.txt             # Python dependencies for inference
└── README.md
```

## Limitations & Model Constraints

This ONNX model is tied to a specific input contract and normalization:

Pre-set history length: input_length = 288 time steps by default.

+ Forecast horizon: forecast_length = H.

+ Signals: CPU utilization-only (AvgCPU, MaxCPU). Other resources (RAM/disk/net) are not modeled in this version.

+ Normalization:

    + CPU values are expected in 0–100 in CSV and normalized by cpu_divisor (usually 100.0).

    + Remaining lifetime is normalized by max_lifetime_seconds from model_config.json.

+ Output ranges: the model outputs are bounded to [0, 1] (Sigmoid heads), so it will not produce values outside this range.

+ Data schema requirement for demo: demodata.csv must contain at least:

    + VMID

    + AvgCPU, MaxCPU

    + time_relative_seconds, lifetime_seconds

    + optionally TimeStamp and MaxCPU_so_far (if missing, demo falls back to last MaxCPU)

## Installation

It is recommended to use a virtual environment to keep dependencies isolated.

### 1. Clone the Repository
```bash
git clone https://github.com/mlsysops-eu/model-peaklife-predictive-vm-management
cd model-peaklife-predictive-vm-management
```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```Bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```Bash
pip install -r requirements.txt
```

## Quick Start
Run the demo script.

```Bash
python demo.py
```

**Output Example:**
```Plaintext
--- Model Loaded ---
--- Demo Dataset Ready ---
--- PeakLife Demo ---
VMID: QdbZeJFmsJ3euIQ4lwW63NwFEP+QIirT4QbI0jEGr4dpkOet8p3iQSHAEm1gKWnR
inputs shape: (1, 288, 2) | aux shape: (1, 2)
pred_util shape: (1, 1, 2) | pred_life shape: (1, 1)

--- Utilization Prediction (AvgCPU, MaxCPU) ---
Pred (normalized): 3.912 , 5.861 | 0.039123 , 0.058608
True (normalized): 4.356 , 6.186 | 0.043559 , 0.061863

--- MAPE (avg, max, and combined) ---
util_mape(avg)=0.101832 | util_mape(max)=0.052609 | util_mape(combined)=0.077221

--- Remaining Lifetime Prediction ---
Pred remaining_lifetime_norm=0.893175 | Pred remaining_lifetime_seconds=1544747.0s
True remaining_lifetime_norm=0.950043 | True remaining_lifetime_seconds=1643100.0s

--- Lifetime MAPE ---
life_mape=0.059858
```

## Configuration & Model Card

The file model/model_config.json serves as the Model Card and includes:

    + input_length

    + forecast_length

    + normalization.cpu_divisor

    + normalization.max_lifetime_seconds

    + input/output names and shapes (if you record them)

**Important**: The ONNX weights are tied to these dimensions and normalization constants.

## Citation

If you use this repository or the provided ONNX model in your research, please cite it.

Citation metadata is available via the **Zenodo DOI** above and in the # PeakLife (ONNX) — VM Utilization + Remaining Lifetime Predictor

This repository contains **PeakLife**, a lightweight neural model exported to **ONNX** for portable inference.  
Given the historic utilization information for a VM, PeakLife predicts:

- **Future CPU utilization**: **AvgCPU** and **MaxCPU** (normalized)
- **Remaining lifetime**: normalized remaining lifetime (and seconds via scaling)

The repo includes a minimal demo pipeline that loads a small CSV (`demodata.csv`), preprocesses it to the model’s expected inputs, runs ONNX inference, and prints the results.

## Project Structure

```text
.
├── model/
│   ├── peaklife.onnx            # ONNX model
│   └── model_config.json        # Model card & Configuration
├── src/ (Optional)              # Helpers for data preparation
│   ├── DataUtil.py        
│   └── prepare_demodata.py     
├── demo.py                      # Main entry point for inference demo
├── demodata.csv                 # Demo dataset
├── requirements.txt             # Python dependencies for inference
└── README.md
```

## Limitations & Model Constraints

This ONNX model is tied to a specific input contract and normalization:

Pre-set history length: input_length = 288 time steps by default.

+ Forecast horizon: forecast_length = H.

+ Signals: CPU utilization-only (AvgCPU, MaxCPU). Other resources (RAM/disk/net) are not modeled in this version.

+ Normalization:

    + CPU values are expected in 0–100 in CSV and normalized by cpu_divisor (usually 100.0).

    + Remaining lifetime is normalized by max_lifetime_seconds from model_config.json.

+ Output ranges: the model outputs are bounded to [0, 1] (Sigmoid heads), so it will not produce values outside this range.

+ Data schema requirement for demo: demodata.csv must contain at least:

    + VMID

    + AvgCPU, MaxCPU

    + time_relative_seconds, lifetime_seconds

    + optionally TimeStamp and MaxCPU_so_far (if missing, demo falls back to last MaxCPU)

## Installation

It is recommended to use a virtual environment to keep dependencies isolated.

### 1. Clone the Repository
```bash
git clone https://github.com/tgasla/MLSysOps-VM-Management-Agent.git
cd MLSysOps-VM-Management-Agent
```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```Bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```Bash
pip install -r requirements.txt
```

## Quick Start
Run the demo script.

```Bash
python demo.py
```

**Output Example:**
```Plaintext
--- Model Loaded ---
--- Demo Dataset Ready ---
--- PeakLife Demo ---
VMID: QdbZeJFmsJ3euIQ4lwW63NwFEP+QIirT4QbI0jEGr4dpkOet8p3iQSHAEm1gKWnR
inputs shape: (1, 288, 2) | aux shape: (1, 2)
pred_util shape: (1, 1, 2) | pred_life shape: (1, 1)

--- Utilization Prediction (AvgCPU, MaxCPU) ---
Pred (normalized): 3.912 , 5.861 | 0.039123 , 0.058608
True (normalized): 4.356 , 6.186 | 0.043559 , 0.061863

--- MAPE (avg, max, and combined) ---
util_mape(avg)=0.101832 | util_mape(max)=0.052609 | util_mape(combined)=0.077221

--- Remaining Lifetime Prediction ---
Pred remaining_lifetime_norm=0.893175 | Pred remaining_lifetime_seconds=1544747.0s
True remaining_lifetime_norm=0.950043 | True remaining_lifetime_seconds=1643100.0s

--- Lifetime MAPE ---
life_mape=0.059858
```

## Configuration & Model Card

The file model/model_config.json serves as the Model Card and includes:

    + input_length

    + forecast_length

    + normalization.cpu_divisor

    + normalization.max_lifetime_seconds

    + input/output names and shapes (if you record them)

**Important**: The ONNX weights are tied to these dimensions and normalization constants.

## Citation

If you use this repository or the provided ONNX model in your research, please cite it.

Citation metadata is available via the **Zenodo DOI: 10.5281/zenodo.18422585** and in the [Citation.cff](Citation.cff) file.

## Acknowledgments & Funding

This repository was developed by **University of Thessaly (UTH)** as part of the **MLSysOps** project.

The MLSysOps consortium consists of **12 European partners** across academia and industry, working together to optimize AI-driven operations in the Cloud-Edge Continuum.

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement **No 101092912**.

Learn more at [mlsysops.eu](https://mlsysops.eu).
