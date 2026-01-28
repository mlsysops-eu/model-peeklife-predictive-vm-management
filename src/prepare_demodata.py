# prepare_demodata.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to original CSV")
    ap.add_argument("--out", default="demodata.csv", help="Output path (default: demodata.csv)")
    ap.add_argument("--input_length", type=int, default=288)
    ap.add_argument("--forecast_length", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_vms", type=int, default=50, help="Number of training VMs")
    ap.add_argument("--rows_per_vm", type=int, default=350, help="Rows to keep per VM (recommended size >= input+forecast )")
    args = ap.parse_args()

    df = pd.read_csv(args.src, index_col=0)

    valid_len = args.input_length + args.forecast_length
    vm_counts = df.groupby("VMID").size()
    valid_vms = vm_counts[vm_counts >= valid_len].index
    df = df[df["VMID"].isin(valid_vms)].copy()

    # sort per VM
    sort_cols = ["VMID"]
    if "TimeStamp" in df.columns:
        sort_cols.append("TimeStamp")
    elif "time_relative_seconds" in df.columns:
        sort_cols.append("time_relative_seconds")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    vms = df["VMID"].unique()
    if len(vms) < 2:
        train_vms = vms
    else:
        train_vms, _ = train_test_split(vms, test_size=0.2, random_state=args.seed) # optional random seed

    train_vms = list(train_vms)[: min(args.num_vms, len(train_vms))]
    out_parts = []
    for vm in train_vms:
        sub = df[df["VMID"] == vm]
        out_parts.append(sub.iloc[: args.rows_per_vm].copy())

    out_df = pd.concat(out_parts, axis=0).reset_index(drop=True)
    out_df.to_csv(args.out, index=True)  # keep index column consistent
    print(f"Wrote {args.out} with {len(out_df)} rows from {len(train_vms)} VMs.")
    print("   Columns:", list(out_df.columns))

if __name__ == "__main__":
    main()
