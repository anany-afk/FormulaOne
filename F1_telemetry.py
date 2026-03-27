import fastf1
import matplotlib.pyplot as plt

circuit=input("Enter the qualifying circuit (Monza) : ")
season=int(input("Enter season: "))

session = fastf1.get_session(season, circuit, "Q")
session.load()

driver1 = input("Enter first driver (ABC) : ")
driver2 = input("Enter second driver (XYZ) : ")

lap1 = session.laps.pick_driver(driver1).pick_fastest()
lap2 = session.laps.pick_driver(driver2).pick_fastest()
tel1 = lap1.get_telemetry()
tel2 = lap2.get_telemetry()


plt.figure()
plt.plot(tel1["Distance"], tel1["Speed"], label=driver1)
plt.plot(tel2["Distance"], tel2["Speed"], label=driver2)
plt.xlabel("Distance (m)")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.title("Speed Comparison – Qualifying Lap")
plt.show()

delta_time = tel1["Time"] - tel2["Time"]

plt.figure()
plt.plot(tel1["Distance"], delta_time.dt.total_seconds())
plt.axhline(0)
plt.xlabel("Distance (m)")
plt.ylabel("Delta Time (s)")
plt.title("Lap Time Delta (Positive = Driver 1 slower)")
plt.show()

plt.figure()
plt.plot(tel1["Distance"], tel1["Throttle"], label=f"{driver1} Throttle")
plt.plot(tel2["Distance"], tel2["Throttle"], label=f"{driver2} Throttle")
plt.xlabel("Distance")
plt.ylabel("Throttle (%)")
plt.legend()
plt.show()
pos1 = lap1.get_pos_data()
pos2 = lap2.get_pos_data()

plt.figure()
plt.plot(pos1["X"], pos1["Y"], label=driver1)
plt.plot(pos2["X"], pos2["Y"], label=driver2)
plt.axis("equal")
plt.legend()
plt.title("Racing Line Comparison")
plt.show()
