from pathlib import Path
import pandas as pd

def import_data(file_path : str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"The file {file_path} is not a CSV file.")
    
    return pd.read_csv(file_path)

def load_retail_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    return {
        "stores": import_data(data_dir / "stores.csv"),
        "train": import_data(data_dir / "train.csv"),
        "oil" : import_data(data_dir / "oil.csv"),
        "holidays_events": import_data(data_dir / "holidays_events.csv"),
        "transactions": import_data(data_dir / "transactions.csv"),
        "test": import_data(data_dir / "test.csv")
    }