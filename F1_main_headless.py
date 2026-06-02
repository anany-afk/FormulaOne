import sys
import fastf1
import os
import logging
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

logging.getLogger("fastf1").setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from F1_segmentation import segment_by_braking
from F1_features import extract_segment_features
from F1_visualization import (
    plot_braking_zones,
    compare_braking_zones,
    plot_model_verdict,
    plot_speed_heatmap,
    plot_time_loss_map,
    plot_braking_distance_markers,
    plot_synchronized_traces,
    plot_corner_cards,
    plot_consistency_dashboard,
)


if len(sys.argv) < 7:
    print("ERROR: Usage: python F1_main_headless.py <circuit> <season> <session> <d1> <d2> <output_dir>")
    sys.exit(1)

circuit      = sys.argv[1].strip().title()
season       = int(sys.argv[2].strip())
session_type = sys.argv[3].strip().upper()
driver1      = sys.argv[4].strip().upper()
driver2      = sys.argv[5].strip().upper()
output_dir   = sys.argv[6].strip()
print(f"[DEBUG] output_dir = {output_dir}")

os.makedirs(output_dir, exist_ok=True)

def save(name):
    path = os.path.abspath(os.path.join(output_dir, name))

    plt.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=plt.gcf().get_facecolor())

    print(f"[SAVED] {path}")
    print(f"[EXISTS] {os.path.exists(path)}")

    plt.close("all")


_orig_show = plt.show
def _patched_show():
    pass
plt.show = _patched_show


print(f"[FastF1] Cache at: {CACHE_DIR}")
print(f"Loading session: {circuit.upper()} {season} {session_type}...")
sys.stdout.flush()

session = fastf1.get_session(season, circuit, session_type)
session.load()
print("[FastF1] Session loaded.")
sys.stdout.flush()


train_laps_1 = session.laps.pick_drivers([driver1]).pick_quicklaps()
train_laps_2 = session.laps.pick_drivers([driver2]).pick_quicklaps()
viz_lap_1    = train_laps_1.pick_fastest()
viz_lap_2    = train_laps_2.pick_fastest()

print(f"Quick laps — {driver1}: {len(train_laps_1)}  |  {driver2}: {len(train_laps_2)}")
sys.stdout.flush()


print("Segmenting by braking zones...")
sys.stdout.flush()

dataset = []
for _, lap in train_laps_1.iterlaps():
    tel = lap.get_telemetry()
    for seg in segment_by_braking(tel):
        row = extract_segment_features(seg, driver1)
        if row:
            dataset.append(row)

for _, lap in train_laps_2.iterlaps():
    tel = lap.get_telemetry()
    for seg in segment_by_braking(tel):
        row = extract_segment_features(seg, driver2)
        if row:
            dataset.append(row)

print(f"Total segments extracted: {len(dataset)}")
sys.stdout.flush()


df = pd.DataFrame(dataset)
df.to_csv(os.path.join(output_dir, "segments_dataset.csv"), index=False)
print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} cols")
sys.stdout.flush()

df["driver_id"] = df["driver"].map({driver1: 0, driver2: 1})
df["faster"]    = (df["driver_id"] == 1).astype(int)
df = df.dropna()
df = df[df["segment_time"] > 0]

features = ["avg_speed", "min_speed", "max_speed", "avg_throttle", "brake_ratio", "segment_length"]
X = df[features]
y = df["faster"]


print("Training RandomForestClassifier...")
sys.stdout.flush()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred))
sys.stdout.flush()


print("Rendering visualizations...")
sys.stdout.flush()

plot_model_verdict(X_test=X_test, y_test=y_test, model=model, driver1=driver1, driver2=driver2)
save("model_verdict.png")

plot_braking_zones(viz_lap_1, driver1, circuit)
save(f"{driver1}_braking_zones.png")

plot_braking_zones(viz_lap_2, driver2, circuit)
save(f"{driver2}_braking_zones.png")

compare_braking_zones(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
save("braking_comparison.png")

plot_speed_heatmap(viz_lap_1, driver1, circuit)
save(f"{driver1}_speed_heatmap.png")

plot_speed_heatmap(viz_lap_2, driver2, circuit)
save(f"{driver2}_speed_heatmap.png")

plot_time_loss_map(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
save("time_loss_map.png")

plot_braking_distance_markers(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
save("braking_distance_markers.png")

plot_synchronized_traces(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
save("synchronized_traces.png")

plot_corner_cards(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
save("corner_cards.png")

plot_consistency_dashboard(train_laps_1, train_laps_2, driver1, driver2, circuit)
save("consistency_dashboard.png")

print("[DONE] All plots saved.")
sys.stdout.flush()