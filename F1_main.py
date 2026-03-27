import fastf1
import os
import logging
logging.getLogger("fastf1").setLevel(logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
print(f"[FastF1] Cache enabled at: {CACHE_DIR}")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from F1_segmentation import segment_by_braking
from F1_features import extract_segment_features
from F1_visualization import plot_model_verdict
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

# Current session

circuit = input("Enter circuit (e.g., Suzuka, Monza): ").strip().title()

season = int(input("Enter season (e.g., 2025): ").strip())

session_type = input("Enter session type (R = Race, Q = Qualifying, FP1/FP2/FP3): ").strip().upper()

driver1 = input("Enter Driver 1 (e.g., VER): ").strip().upper()
driver2 = input("Enter Driver 2 (e.g., HAM): ").strip().upper()

# Load session
session = fastf1.get_session(season, circuit, session_type)
session.load()

# Training laps
train_laps_1 = session.laps.pick_drivers([driver1]).pick_quicklaps()
train_laps_2 = session.laps.pick_drivers([driver2]).pick_quicklaps()

# Visualization laps
viz_lap_1 = train_laps_1.pick_fastest()
viz_lap_2 = train_laps_2.pick_fastest()


# ML training data

all_segments = []

# Driver 1
for _, lap in train_laps_1.iterlaps():
    tel = lap.get_telemetry()
    segments = segment_by_braking(tel)
    all_segments.extend(segments)

# Driver 2
for _, lap in train_laps_2.iterlaps():
    tel = lap.get_telemetry()
    segments = segment_by_braking(tel)
    all_segments.extend(segments)

print(f"Total segments extracted: {len(all_segments)}")


#dataset creation

dataset = []

for _, lap in train_laps_1.iterlaps():
    tel = lap.get_telemetry()
    segments = segment_by_braking(tel)

    for seg in segments:
        row = extract_segment_features(seg, driver1)
        if row:
            dataset.append(row)

for _, lap in train_laps_2.iterlaps():
    tel = lap.get_telemetry()
    segments = segment_by_braking(tel)

    for seg in segments:
        row = extract_segment_features(seg, driver2)
        if row:
            dataset.append(row)
len(dataset)  

df = pd.read_csv("segments_dataset.csv")
print("Dataset loaded.")

#check for daaset successful creation
df = pd.DataFrame(dataset)
print(df.shape)
print(df.head())


#FOR NEW DATASET ONLY
df.to_csv("segments_dataset.csv", index=False)
print("Dataset saved.")


# label encoding (driver)
df["driver_id"] = df["driver"].map({
    driver1: 0,
    driver2: 1
})

# target (faster)
df["faster"] = (df["driver_id"] == 1).astype(int)

#cleaning
df = df.dropna()
df = df[df["segment_time"] > 0]

features = [
    "avg_speed",
    "min_speed",
    "max_speed",
    "avg_throttle",
    "brake_ratio",
    "segment_length"
]

X = df[features]
y = df["faster"]


#train,test,split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#simple model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


#accuracy evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



"""sanity check
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()"""

#visualization
plot_model_verdict(
    X_test=X_test,
    y_test=y_test,
    model=model,
    driver1=driver1,
    driver2=driver2
)
plot_braking_zones(viz_lap_1, driver1, circuit)
plot_braking_zones(viz_lap_2, driver2, circuit)
compare_braking_zones(viz_lap_1, viz_lap_2, driver1, driver2, circuit)

plot_speed_heatmap(viz_lap_1, driver1, circuit)
plot_speed_heatmap(viz_lap_2, driver2, circuit)
plot_time_loss_map(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
plot_braking_distance_markers(viz_lap_1, viz_lap_2, driver1, driver2, circuit)

plot_synchronized_traces(viz_lap_1, viz_lap_2, driver1, driver2, circuit)
plot_corner_cards(viz_lap_1, viz_lap_2, driver1, driver2, circuit)

plot_consistency_dashboard(train_laps_1, train_laps_2, driver1, driver2, circuit)