import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d


def plot_braking_zones(lap, driver, circuit):
    tel = lap.get_telemetry()

    brake_on = tel["Brake"] == True
    brake_start = brake_on & (~brake_on.shift(1, fill_value=False))
    braking_points = tel.loc[brake_start]

    plt.figure(figsize=(8, 6))
    plt.plot(tel["X"], tel["Y"], color="lightgray", label="Racing Line")
    plt.scatter(
        braking_points["X"],
        braking_points["Y"],
        color="red", s=30, label="Braking Start"
    )
    plt.title(f"Braking Zones – {driver} | {circuit}")
    plt.axis("equal")
    plt.legend()
    plt.show()


def compare_braking_zones(lap1, lap2, d1, d2, circuit):
    tel1 = lap1.get_telemetry()
    tel2 = lap2.get_telemetry()

    b1 = (tel1["Brake"] == True) & (~(tel1["Brake"] == True).shift(1, fill_value=False))
    b2 = (tel2["Brake"] == True) & (~(tel2["Brake"] == True).shift(1, fill_value=False))

    plt.figure(figsize=(8, 6))
    plt.plot(tel1["X"], tel1["Y"], color="lightgray")
    plt.scatter(tel1.loc[b1, "X"], tel1.loc[b1, "Y"], color="red",  label=d1)
    plt.scatter(tel2.loc[b2, "X"], tel2.loc[b2, "Y"], color="blue", label=d2)
    plt.title(f"Braking Zone Comparison | {circuit}")
    plt.axis("equal")
    plt.legend()
    plt.show()


def plot_model_verdict(X_test, y_test, model, driver1, driver2):
    y_pred = model.predict(X_test)
    segment_index = range(len(y_pred))

    plt.figure(figsize=(10, 4))
    plt.scatter(segment_index, y_pred, c=y_pred, cmap="coolwarm", s=40, alpha=0.8)
    plt.yticks([0, 1], [driver1, driver2])
    plt.xlabel("Track Segment Index")
    plt.ylabel("Predicted Faster Driver")
    plt.title("Model Verdict Across Track Segments")
    plt.grid(True, axis="x", alpha=0.3)
    plt.show()


def plot_speed_heatmap(lap, driver, circuit):
    tel = lap.get_telemetry()
    tel = tel.dropna(subset=["X", "Y", "Speed"])

    x = tel["X"].values
    y = tel["Y"].values
    speed = tel["Speed"].values

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    speed_mid = (speed[:-1] + speed[1:]) / 2

    norm = plt.Normalize(speed_mid.min(), speed_mid.max())
    lc = LineCollection(segments, cmap="RdYlGn", norm=norm, linewidth=3, alpha=0.9)
    lc.set_array(speed_mid)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 100, x.max() + 100)
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = plt.colorbar(lc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Speed (km/h)", fontsize=10)
    ax.set_title(f"Speed Heatmap — {driver} | {circuit}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_time_loss_map(lap1, lap2, driver1, driver2, circuit):
    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()

    tel1 = tel1.dropna(subset=["Distance", "Time", "X", "Y"])
    tel2 = tel2.dropna(subset=["Distance", "Time"])

    ref_dist = tel1["Distance"].values
    t1 = tel1["Time"].dt.total_seconds().values
    t2 = np.interp(ref_dist, tel2["Distance"].values, tel2["Time"].dt.total_seconds().values)
    delta = t2 - t1

    x = tel1["X"].values
    y = tel1["Y"].values

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    delta_mid = (delta[:-1] + delta[1:]) / 2

    abs_max = np.percentile(np.abs(delta_mid), 95)
    norm = plt.Normalize(-abs_max, abs_max)
    lc = LineCollection(segments, cmap="coolwarm_r", norm=norm, linewidth=4, alpha=0.95)
    lc.set_array(delta_mid)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 100, x.max() + 100)
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = plt.colorbar(lc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(f"← {driver1} faster  |  {driver2} faster →", fontsize=9)
    ax.set_title(
        f"Time Loss Map — {driver1} vs {driver2} | {circuit}\n"
        f"Blue = {driver1} gaining | Red = {driver2} gaining",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def plot_braking_distance_markers(lap1, lap2, driver1, driver2, circuit, n_corners=10):
    def get_brake_events(lap):
        tel = lap.get_telemetry().add_distance()
        tel = tel.dropna(subset=["X", "Y", "Distance", "Brake"])
        brake_on = tel["Brake"] == True
        brake_start = brake_on & (~brake_on.shift(1, fill_value=False))
        return tel.loc[brake_start].reset_index(drop=True)

    bp1 = get_brake_events(lap1)
    bp2 = get_brake_events(lap2)
    tel1 = lap1.get_telemetry().dropna(subset=["X", "Y"])

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(tel1["X"], tel1["Y"], color="#2a2a2a", linewidth=1.5, alpha=0.4, zorder=1)

    matched, used = [], set()
    for _, row1 in bp1.iterrows():
        dists = np.abs(bp2["Distance"].values - row1["Distance"])
        idx = dists.argmin()
        if idx not in used and dists[idx] < 300:
            matched.append((row1, bp2.iloc[idx]))
            used.add(idx)

    for i, (b1, b2) in enumerate(matched[:n_corners]):
        diff = b2["Distance"] - b1["Distance"]
        mx = (b1["X"] + b2["X"]) / 2
        my = (b1["Y"] + b2["Y"]) / 2

        ax.scatter(b1["X"], b1["Y"], color="red",  s=50, zorder=3)
        ax.scatter(b2["X"], b2["Y"], color="blue", s=50, zorder=3)

        label = f"+{diff:.0f}m {driver2}" if diff > 1 else (
                f"+{-diff:.0f}m {driver1}" if diff < -1 else "≈ same")
        color = "blue" if diff > 1 else ("red" if diff < -1 else "gray")

        ax.annotate(label, xy=(mx, my), fontsize=7.5, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85), zorder=5)

    red_patch  = mpatches.Patch(color="red",  label=f"{driver1} braking point")
    blue_patch = mpatches.Patch(color="blue", label=f"{driver2} braking point")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Braking Point Distance Markers — {driver1} vs {driver2} | {circuit}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def plot_synchronized_traces(lap1, lap2, driver1, driver2, circuit):
    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()

    tel1 = tel1.dropna(subset=["Distance", "Speed", "Throttle", "Brake"])
    tel2 = tel2.dropna(subset=["Distance", "Speed", "Throttle", "Brake"])

    colors = {driver1: "#e8002d", driver2: "#0067ff"}

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Synchronized Telemetry — {driver1} vs {driver2} | {circuit}",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = GridSpec(3, 1, hspace=0.08)

    ax_speed = fig.add_subplot(gs[0])
    ax_thr   = fig.add_subplot(gs[1], sharex=ax_speed)
    ax_brake = fig.add_subplot(gs[2], sharex=ax_speed)

    ax_speed.plot(tel1["Distance"], tel1["Speed"], color=colors[driver1], lw=1.4, label=driver1)
    ax_speed.plot(tel2["Distance"], tel2["Speed"], color=colors[driver2], lw=1.4, label=driver2, alpha=0.85)
    ax_speed.set_ylabel("Speed\n(km/h)", fontsize=9)
    ax_speed.legend(loc="upper right", fontsize=9)
    ax_speed.grid(True, alpha=0.2)
    plt.setp(ax_speed.get_xticklabels(), visible=False)

    ax_thr.plot(tel1["Distance"], tel1["Throttle"], color=colors[driver1], lw=1.2)
    ax_thr.plot(tel2["Distance"], tel2["Throttle"], color=colors[driver2], lw=1.2, alpha=0.85)
    ax_thr.set_ylabel("Throttle\n(%)", fontsize=9)
    ax_thr.set_ylim(-5, 105)
    ax_thr.grid(True, alpha=0.2)
    plt.setp(ax_thr.get_xticklabels(), visible=False)

    brake1 = tel1["Brake"].astype(float)
    brake2 = tel2["Brake"].astype(float)
    ax_brake.fill_between(tel1["Distance"], brake1, alpha=0.5, color=colors[driver1], label=driver1)
    ax_brake.fill_between(tel2["Distance"], brake2 * 0.6, alpha=0.4, color=colors[driver2], label=driver2)
    ax_brake.set_ylabel("Brake\n(on/off)", fontsize=9)
    ax_brake.set_xlabel("Distance (m)", fontsize=10)
    ax_brake.set_ylim(-0.05, 1.2)
    ax_brake.set_yticks([0, 1])
    ax_brake.set_yticklabels(["Off", "On"])
    ax_brake.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_corner_cards(lap1, lap2, driver1, driver2, circuit, n_corners=8):
    def get_corner_stats(lap):
        tel = lap.get_telemetry().add_distance()
        tel = tel.dropna(subset=["Distance", "Speed", "Throttle", "Brake"])
        brake_on = tel["Brake"] == True
        brake_start = brake_on & (~brake_on.shift(1, fill_value=False))
        starts = tel.index[brake_start].tolist()
        stats = []
        for i in range(len(starts) - 1):
            seg = tel.loc[starts[i]:starts[i+1]]
            if len(seg) < 5:
                continue
            stats.append({
                "brake_dist":     tel.loc[starts[i], "Distance"],
                "min_speed":      seg["Speed"].min(),
                "exit_throttle":  seg["Throttle"].iloc[int(len(seg) * 0.6):].mean(),
            })
        return stats

    s1 = get_corner_stats(lap1)
    s2 = get_corner_stats(lap2)
    n  = min(n_corners, len(s1), len(s2))

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    fig.suptitle(f"Corner Cards — {driver1} vs {driver2} | {circuit}", fontsize=13, fontweight="bold")
    axes = axes.flatten() if n > 1 else [axes]

    c1, c2 = "#e8002d", "#0067ff"

    for i in range(n):
        ax = axes[i]
        a, b = s1[i], s2[i]

        def winner_colors(diff):
            if abs(diff) < 0.5:
                return "gray", "gray"
            return (c2, "lightgray") if diff > 0 else ("lightgray", c1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 0.92, f"Corner {i+1}", ha="center", fontsize=10,
                fontweight="bold", transform=ax.transAxes)

        brake_diff = b["brake_dist"] - a["brake_dist"]
        wc2, wc1 = winner_colors(brake_diff)
        ax.text(0.5, 0.74, "Brakes Later",    ha="center", fontsize=8, color="dimgray", transform=ax.transAxes)
        ax.text(0.25, 0.60, f"{driver1}\n{a['brake_dist']:.0f}m", ha="center", fontsize=8, color=wc1, fontweight="bold", transform=ax.transAxes)
        ax.text(0.75, 0.60, f"{driver2}\n{b['brake_dist']:.0f}m", ha="center", fontsize=8, color=wc2, fontweight="bold", transform=ax.transAxes)

        speed_diff = b["min_speed"] - a["min_speed"]
        sc2, sc1 = winner_colors(speed_diff)
        ax.text(0.5, 0.44, "Min Corner Speed", ha="center", fontsize=8, color="dimgray", transform=ax.transAxes)
        ax.text(0.25, 0.32, f"{a['min_speed']:.0f} km/h", ha="center", fontsize=8, color=sc1, fontweight="bold", transform=ax.transAxes)
        ax.text(0.75, 0.32, f"{b['min_speed']:.0f} km/h", ha="center", fontsize=8, color=sc2, fontweight="bold", transform=ax.transAxes)

        thr_diff = b["exit_throttle"] - a["exit_throttle"]
        tc2, tc1 = winner_colors(thr_diff)
        ax.text(0.5, 0.18, "Exit Throttle",   ha="center", fontsize=8, color="dimgray", transform=ax.transAxes)
        ax.text(0.25, 0.06, f"{a['exit_throttle']:.1f}%", ha="center", fontsize=8, color=tc1, fontweight="bold", transform=ax.transAxes)
        ax.text(0.75, 0.06, f"{b['exit_throttle']:.1f}%", ha="center", fontsize=8, color=tc2, fontweight="bold", transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_consistency_dashboard(laps1, laps2, driver1, driver2, circuit):
    def get_traces(laps):
        all_speed, all_throttle, all_brake = [], [], []
        ref_dist = None

        for _, lap in laps.iterlaps():
            try:
                tel = lap.get_telemetry().add_distance()
                tel = tel.dropna(subset=["Distance", "Speed", "Throttle", "Brake"])
                tel = tel.drop_duplicates(subset=["Distance"])
                tel = tel.sort_values("Distance")
                if len(tel) < 20:
                    continue
                if ref_dist is None:
                    ref_dist = np.linspace(tel["Distance"].min(), tel["Distance"].max(), 500)
                f_speed = interp1d(tel["Distance"], tel["Speed"],              bounds_error=False, fill_value="extrapolate")
                f_thr   = interp1d(tel["Distance"], tel["Throttle"],           bounds_error=False, fill_value="extrapolate")
                f_brake = interp1d(tel["Distance"], tel["Brake"].astype(float),bounds_error=False, fill_value="extrapolate")
                all_speed.append(np.nan_to_num(f_speed(ref_dist)))
                all_throttle.append(np.nan_to_num(f_thr(ref_dist)))
                all_brake.append(np.nan_to_num(f_brake(ref_dist)))
            except Exception:
                continue

        return ref_dist, np.array(all_speed), np.array(all_throttle), np.array(all_brake)

    def consistency_score(arr):
        n_laps = len(arr)
        if n_laps < 3:
            return np.std(arr, axis=0).mean()
        detrended = np.zeros_like(arr)
        for point_idx in range(arr.shape[1]):
            col      = arr[:, point_idx]
            lap_nums = np.arange(n_laps)
            trend    = np.polyfit(lap_nums, col, deg=1)
            correction = np.polyval(trend, lap_nums)
            detrended[:, point_idx] = col - correction + col.mean()
        return np.std(detrended, axis=0).mean()

    dist1, spd1, thr1, brk1 = get_traces(laps1)
    dist2, spd2, thr2, brk2 = get_traces(laps2)

    if dist1 is None or dist2 is None:
        print("Not enough telemetry data for consistency dashboard.")
        return

    scores = {
        driver1: {"speed": consistency_score(spd1), "throttle": consistency_score(thr1), "brake": consistency_score(brk1)},
        driver2: {"speed": consistency_score(spd2), "throttle": consistency_score(thr2), "brake": consistency_score(brk2)},
    }

    overall1 = np.mean(list(scores[driver1].values()))
    overall2 = np.mean(list(scores[driver2].values()))
    verdict  = driver1 if overall1 < overall2 else driver2

    c1, c2 = "#e8002d", "#0067ff"
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(f"Consistency Report — {driver1} vs {driver2}  |  {circuit}",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.12,
                  top=0.93, bottom=0.07, left=0.07, right=0.97)

    def style_ax(ax):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="gray", labelsize=8)
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    for col, (driver, dist, spd, color) in enumerate([
        (driver1, dist1, spd1, c1),
        (driver2, dist2, spd2, c2),
    ]):
        ax = fig.add_subplot(gs[0, col])
        mean_spd = spd.mean(axis=0)
        std_spd  = spd.std(axis=0)
        for trace in spd:
            ax.plot(dist, trace, color=color, alpha=0.12, lw=0.8)
        ax.plot(dist, mean_spd, color=color, lw=2, label="Mean")
        ax.fill_between(dist, mean_spd - std_spd, mean_spd + std_spd, color=color, alpha=0.15, label="±1 std")
        ax.set_title(f"{driver} — Speed", fontsize=10, fontweight="bold")
        ax.set_ylabel("km/h", fontsize=8)
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        style_ax(ax)

    for col, (driver, dist, thr, color) in enumerate([
        (driver1, dist1, thr1, c1),
        (driver2, dist2, thr2, c2),
    ]):
        ax = fig.add_subplot(gs[1, col])
        mean_thr = thr.mean(axis=0)
        std_thr  = thr.std(axis=0)
        for trace in thr:
            ax.plot(dist, trace, color=color, alpha=0.12, lw=0.8)
        ax.plot(dist, mean_thr, color=color, lw=2)
        ax.fill_between(dist, mean_thr - std_thr, mean_thr + std_thr, color=color, alpha=0.15)
        ax.set_title(f"{driver} — Throttle", fontsize=10, fontweight="bold")
        ax.set_ylabel("%", fontsize=8)
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.set_ylim(-5, 110)
        style_ax(ax)

    for col, (driver, dist, brk, color) in enumerate([
        (driver1, dist1, brk1, c1),
        (driver2, dist2, brk2, c2),
    ]):
        ax = fig.add_subplot(gs[2, col])
        mean_brk = brk.mean(axis=0)
        for trace in brk:
            ax.plot(dist, trace, color=color, alpha=0.10, lw=0.8)
        ax.plot(dist, mean_brk, color=color, lw=2)
        ax.set_title(f"{driver} — Brake", fontsize=10, fontweight="bold")
        ax.set_ylabel("Brake (0/1)", fontsize=8)
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.set_ylim(-0.05, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Off", "On"], color="gray", fontsize=7)
        style_ax(ax)

    ax_bars    = fig.add_subplot(gs[3, 0])
    ax_verdict = fig.add_subplot(gs[3, 1])

    metrics = ["speed", "throttle", "brake"]
    x = np.arange(len(metrics))
    w = 0.3
    vals1   = [scores[driver1][m] for m in metrics]
    vals2   = [scores[driver2][m] for m in metrics]
    max_val = max(max(vals1), max(vals2)) + 0.1
    inv1    = [max_val - v for v in vals1]
    inv2    = [max_val - v for v in vals2]

    ax_bars.bar(x - w/2, inv1, width=w, color=c1, alpha=0.85, label=driver1)
    ax_bars.bar(x + w/2, inv2, width=w, color=c2, alpha=0.85, label=driver2)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(["Speed", "Throttle", "Brake"], color="gray", fontsize=9)
    ax_bars.set_title("Consistency Score by Metric\n(higher = more consistent)", fontsize=9, fontweight="bold")
    ax_bars.legend(fontsize=8)
    ax_bars.set_yticks([])
    style_ax(ax_bars)

    ax_verdict.axis("off")
    ax_verdict.set_facecolor("#1a1a1a")
    verdict_color = c1 if verdict == driver1 else c2
    margin = abs(overall1 - overall2) / max(overall1, overall2) * 100

    ax_verdict.text(0.5, 0.72, "MOST CONSISTENT DRIVER", ha="center",
                    fontsize=9, color="gray", transform=ax_verdict.transAxes)
    ax_verdict.text(0.5, 0.50, verdict, ha="center",
                    fontsize=38, fontweight="bold", color=verdict_color, transform=ax_verdict.transAxes)
    ax_verdict.text(0.5, 0.28, f"{margin:.1f}% more consistent overall", ha="center",
                    fontsize=10, color="white", transform=ax_verdict.transAxes)

    for i, m in enumerate(metrics):
        winner = driver1 if scores[driver1][m] < scores[driver2][m] else driver2
        wcolor = c1 if winner == driver1 else c2
        y_pos  = 0.14 - i * 0.065
        ax_verdict.text(0.35, y_pos, f"{m.capitalize()}:", ha="right", fontsize=8,
                        color="gray", transform=ax_verdict.transAxes)
        ax_verdict.text(0.38, y_pos, winner, ha="left", fontsize=8,
                        color=wcolor, fontweight="bold", transform=ax_verdict.transAxes)

    plt.show()