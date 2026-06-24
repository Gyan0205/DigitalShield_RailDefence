import json

data = json.load(open("dataset/metadata/video_metadata.json"))
dates = [(r["date"], r["day_of_week"], r["platform_number"]) for r in data["records"]]

print(f"{'Date':<14} {'Day':<12} {'PF':<5} {'Status'}")
print("-" * 40)
all_ok = True
for d, day, pf in sorted(dates):
    in_range = "2026-01-01" <= d <= "2026-03-27"
    status = "OK" if in_range else "OUT OF RANGE"
    if not in_range:
        all_ok = False
    print(f"{d:<14} {day:<12} {str(pf):<5} {status}")

print()
print(f"Total records : {len(dates)}")
print(f"All in range  : {all_ok}")
