import re

directory_raw = """sub-example0001  sub-example0002  sub-example0003  sub-example0005
sub-example0006  sub-example0008  sub-example0009  sub-example0010
"""

csv_raw = """example0001
example0002
example0004
example0005
example0006
example0009
example0010
example0011
example0014
"""

dir_ids = set(re.findall(r'sub-(\w+)', directory_raw))                              #extract IDs from directory (strip "sub-" prefix)
csv_ids = set(line.strip() for line in csv_raw.strip().split('\n') if line.strip()) #extract IDs from CSV
only_in_dir = sorted(dir_ids - csv_ids)
only_in_csv = sorted(csv_ids - dir_ids)
in_both = dir_ids & csv_ids

print(f"Total in directory: {len(dir_ids)}")
print(f"Total in CSV: {len(csv_ids)}")
print(f"In both: {len(in_both)}")
print(f"\nOnly in DIRECTORY (not in CSV): {len(only_in_dir)}")
for s in only_in_dir:
    print(f"  {s}")
print(f"\nOnly in CSV (not in directory): {len(only_in_csv)}")
for s in only_in_csv:
    print(f"  {s}")