import csv
import math

input_file = "data/bindingdb_affinity_kd.csv"
output_file = "data/bindingdb_affinity_pkd.csv"

def parse_affinity(value):
    try:
        # Remove any non-numeric characters (like '>', '<') and convert to float
        return float(value.replace(">", "").replace("<", "").strip())
    except Exception:
        return None

with open(input_file, "r", newline='', encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["compound_iso_smiles", "target_sequence", "pKd"])
    for row in reader:
        smiles = row["compound_iso_smiles"]
        target = row["target_sequence"]
        affinity_nM = parse_affinity(row["affinity"])
        if affinity_nM is not None and affinity_nM > 0:
            pkd = 9 - math.log10(affinity_nM)
            writer.writerow([smiles, target, f"{pkd:.3f}"])
        else:
            writer.writerow([smiles, target, ""])