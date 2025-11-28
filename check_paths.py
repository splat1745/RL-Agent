import os

paths_to_check = [
    r"T:\Auto-Farmer-Data\dataset\dataset1\train",
    r"T:\Auto-Farmer-Data\dataset\dataset1\valid",
    r"T:\Auto-Farmer-Data\dataset\dataset2\train",
    r"T:\Auto-Farmer-Data\dataset\dataset2\valid",
]

for p in paths_to_check:
    print(f"Checking {p}...")
    if os.path.exists(p):
        try:
            items = os.listdir(p)
            print(f"  Found {len(items)} items: {items[:5]}...")
        except Exception as e:
            print(f"  Error listing: {e}")
    else:
        print("  DOES NOT EXIST")
