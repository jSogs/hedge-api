import json
import random
import os

# Path to your downloaded JSON file
FILE_PATH = "/Users/julianasogwa/Downloads/api-response.json"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "seriesinDB.txt")

with open(FILE_PATH, "r") as f:
    data = json.load(f)

# Get all series
all_series = data.get('series', [])
print(f"Total series available: {len(all_series)}")

# Pick 100 random series (or all if less than 100)
num_to_pick = min(100, len(all_series))
random_series = random.sample(all_series, num_to_pick)

print(f"Selected {num_to_pick} random series")

# Prepare outputs
output_lines = []

# Format 1: Just tickers
output_lines.append("# Format 1: Just series tickers (for Python list)")
output_lines.append("# Copy this into SERIES = [...]")
output_lines.append("")
for series in random_series:
    ticker = series.get("ticker")
    if ticker:
        output_lines.append(f'"{ticker}",')

output_lines.append("")
output_lines.append("=" * 60)
output_lines.append("")

# Format 2: Ticker with category
output_lines.append("# Format 2: Ticker:Category mapping (for SERIES_CATEGORY dict)")
output_lines.append("# Copy this into SERIES_CATEGORY = {...}")
output_lines.append("")
for series in random_series:
    ticker = series.get("ticker")
    category = series.get("category", "uncategorized")
    if ticker:
        output_lines.append(f'"{ticker}": "{category}",')

# Write to file
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(output_lines))

print(f"\n✓ Wrote {num_to_pick} series to {OUTPUT_FILE}")
print("\nPreview:")
print("=" * 60)
print("\n".join(output_lines[:10]))
print("...")
print(f"\nFull output in: {OUTPUT_FILE}")
