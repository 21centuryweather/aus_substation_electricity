import pandas as pd
from pathlib import Path

def rename_ausnet_raw_files(
    input_dir="/g/data/ng72/pv3484/substation_data/raw_data/VIC_demand/AusNet_VIC",
    output_dir="/g/data/ng72/pv3484/substation_data/raw_data/VIC_demand/AusNet_renamed"
):

    print("Looking for files in:", input_dir)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.csv"))
    print("Found files:", files)

    def extract_ausnet_name(col):
        parts = col.split("\\")
        if len(parts) >= 2:
            name = parts[1]
        else:
            name = col
        return name.replace("'", "").strip()

    def generate_id(name):
        words = name.split()
        if len(words) > 1:
            return f"{words[0][:2].upper()}_{words[1][:2].upper()}"
        return name[:5].upper()

    for csv_file in files:
        print("Processing:", csv_file.name)

        df = pd.read_csv(csv_file)

        meta_cols = df.columns[:3]
        subst_cols = df.columns[3:]

        clean_names = [extract_ausnet_name(c) for c in subst_cols]
        new_ids = [generate_id(n) for n in clean_names]

        assert len(new_ids) == len(set(new_ids)), f"Duplicate IDs in {csv_file.name}"

        df.columns = list(meta_cols) + new_ids

        out_path = output_dir / csv_file.name.replace(".csv", "_renamed.csv")
        df.to_csv(out_path, index=False)

        print("Saved renamed file →", out_path)


if __name__ == "__main__":
    rename_ausnet_raw_files()
