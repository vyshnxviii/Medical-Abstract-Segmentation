import pandas as pd


def parse_pubmed_rct(filepath):
    """Parses the PubMed-RCT format into a list of dicts with abstract_id,
    line_number, total_lines, target, and text."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    abstract_id = None
    abstract_lines = []

    def flush(abstract_id, abstract_lines):
        total = len(abstract_lines)
        for i, (label, text) in enumerate(abstract_lines):
            records.append({
                "abstract_id": abstract_id,
                "line_number": i,
                "total_lines": total,
                "target": label,
                "text": text,
            })

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("###"):
            if abstract_id is not None and abstract_lines:
                flush(abstract_id, abstract_lines)
            abstract_id = line.replace("###", "")
            abstract_lines = []
        elif line.strip() == "":
            continue
        else:
            parts = line.split("\t")
            if len(parts) == 2:
                label, text = parts
                abstract_lines.append((label, text))

    if abstract_id is not None and abstract_lines:
        flush(abstract_id, abstract_lines)

    return pd.DataFrame(records)


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        df = parse_pubmed_rct(f"pubmed-rct/PubMed_20k_RCT/{split}.txt")
        df.to_csv(f"{split}_parsed.csv", index=False)
        print(split, df.shape)
        print(df["target"].value_counts())
