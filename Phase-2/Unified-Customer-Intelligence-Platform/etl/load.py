from pathlib import Path

def save_to_stage(df, stage: str, name: str):
    out_dir = Path(f"data/{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index = False)

    return out_path
