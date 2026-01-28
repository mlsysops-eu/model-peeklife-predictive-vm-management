import json
import numpy as np
import pandas as pd

import onnxruntime as ort

def load_config(path="model/model_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def pick_vm_window(df, input_length, forecast_length):
    sort_cols = ["VMID"]
    if "TimeStamp" in df.columns:
        sort_cols.append("TimeStamp")
    elif "time_relative_seconds" in df.columns:
        sort_cols.append("time_relative_seconds")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    counts = df.groupby("VMID").size()
    candidates = counts[counts >= (input_length + forecast_length)].index.tolist()
    if not candidates:
        raise ValueError("No VMID has enough rows for input_length + forecast_length.")

    vm = candidates[0]
    sub = df[df["VMID"] == vm].reset_index(drop=True)

    start = 0
    hist = sub.iloc[start:start + input_length]
    fut  = sub.iloc[start + input_length:start + input_length + forecast_length]
    return vm, hist, fut

def preprocess(hist, cfg):
    cpu_div = float(cfg["normalization"]["cpu_divisor"])
    max_life = float(cfg["normalization"]["max_lifetime_seconds"])
    max_life = max(max_life, 1e-6)

    avg = (hist["AvgCPU"].to_numpy(dtype=np.float32) / cpu_div).reshape(1, -1, 1)
    mx  = (hist["MaxCPU"].to_numpy(dtype=np.float32) / cpu_div).reshape(1, -1, 1)
    inputs = np.concatenate([avg, mx], axis=-1).astype(np.float32)  # (1,T,2)

    if "MaxCPU_so_far" in hist.columns:
        max_so_far = float(hist["MaxCPU_so_far"].iloc[-1]) / cpu_div
    else:
        max_so_far = float(hist["MaxCPU"].iloc[-1]) / cpu_div

    running_s = float(hist["time_relative_seconds"].iloc[-1])
    aux = np.array([[running_s / max_life, max_so_far]], dtype=np.float32) 
    return inputs, aux

def get_future_targets(fut, cfg):
    cpu_div = float(cfg["normalization"]["cpu_divisor"])
    avg = (fut["AvgCPU"].to_numpy(dtype=np.float32) / cpu_div).reshape(1, -1, 1)
    max  = (fut["MaxCPU"].to_numpy(dtype=np.float32) / cpu_div).reshape(1, -1, 1)
    return np.concatenate([avg, max], axis=-1).astype(np.float32)  # shape (1,H,2)

def mape_floor(pred, target, floor):
    denom = np.maximum(np.abs(target), floor)
    return np.mean(np.abs(pred - target) / denom)

def get_true_remaining_lifetime_norm(fut, cfg):
    max_life = float(cfg["normalization"]["max_lifetime_seconds"])
    max_life = max(max_life, 1e-6)

    row = fut.iloc[-1]
    total_life_s = float(row["lifetime_seconds"])
    running_s    = float(row["time_relative_seconds"])
    remaining_s  = max(0.0, total_life_s - running_s)

    return np.float32(remaining_s / max_life), remaining_s

def main():
    cfg = load_config("model/model_config.json")
    input_length = int(cfg["input_length"])
    forecast_length = int(cfg["forecast_length"])
    print("--- Model Loaded ---")

    df = pd.read_csv("demodata.csv", index_col=0)

    print("--- Demo Dataset Ready ---")

    vm, hist, fut = pick_vm_window(df, input_length, forecast_length)
    x, aux = preprocess(hist, cfg)
    y_true = get_future_targets(fut, cfg)

    sess = ort.InferenceSession("model/peaklife.onnx", providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    feed = {in_names[0]: x, in_names[1]: aux} if len(in_names) == 2 else {in_names[0]: x}
    outs = sess.run(out_names, feed)

    out_map = {name: arr for name, arr in zip(out_names, outs)}
    pred_util = out_map.get("pred_util", outs[0])
    pred_life = out_map.get("pred_life", outs[1] if len(outs) > 1 else None)

    pred_util = np.asarray(pred_util, dtype=np.float32)
    pred_life = np.asarray(pred_life, dtype=np.float32) if pred_life is not None else None

    print("--- PeakLife Demo ---")
    print(f"VMID: {vm}")
    print(f"inputs shape: {x.shape} | aux shape: {aux.shape}")
    print(f"pred_util shape: {pred_util.shape} | pred_life shape: {None if pred_life is None else pred_life.shape}")

    cpu_div = float(cfg["normalization"]["cpu_divisor"])
    pred_util_pct = pred_util * cpu_div
    y_true_pct = y_true * cpu_div

    print("\n--- Utilization Prediction (AvgCPU, MaxCPU) ---")
    print(f"Pred (normalized): {pred_util_pct[0, 0, 0]:.3f} , {pred_util_pct[0, 0, 1]:.3f} | "
          f"{pred_util[0, 0, 0]:.6f} , {pred_util[0, 0, 1]:.6f}")
    print(f"True (normalized): {y_true_pct[0, 0, 0]:.3f} , {y_true_pct[0, 0, 1]:.3f} | "
          f"{y_true[0, 0, 0]:.6f} , {y_true[0, 0, 1]:.6f}")

    floor_util = 0.03
    avg_m = mape_floor(pred_util[:, :, 0], y_true[:, :, 0], floor_util)
    max_m = mape_floor(pred_util[:, :, 1], y_true[:, :, 1], floor_util)
    mean_m = 0.5 * (avg_m + max_m)

    print("\n--- MAPE (avg, max, and combined) ---")
    print(f"util_mape(avg)={avg_m:.6f} | util_mape(max)={max_m:.6f} | util_mape(combined)={mean_m:.6f}")

    if pred_life is not None:
        max_life = float(cfg["normalization"]["max_lifetime_seconds"])
        pred_life_norm = float(pred_life.reshape(-1)[0])
        pred_life_s = pred_life_norm * max_life

        true_life_norm, true_life_s = get_true_remaining_lifetime_norm(fut, cfg)


        floor_life = 0.05
        life_mape = mape_floor(np.array([pred_life_norm], dtype=np.float32),
                               np.array([true_life_norm], dtype=np.float32),
                               floor_life)

        print("\n--- Remaining Lifetime Prediction ---")
        print(f"Pred remaining_lifetime_norm={pred_life_norm:.6f} | Pred remaining_lifetime_seconds={pred_life_s:.1f}s")
        print(f"True remaining_lifetime_norm={true_life_norm:.6f} | True remaining_lifetime_seconds={true_life_s:.1f}s")
        print("\n--- Lifetime MAPE ---")
        print(f"life_mape={life_mape:.6f}")

if __name__ == "__main__":
    main()
