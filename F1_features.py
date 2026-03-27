

def extract_segment_features(segment, driver):
    #converts tel data into one ML row i.e features (dictionary)

    # error check
    if len(segment) < 10:
        return None

    features = {
        "avg_speed": segment["Speed"].mean(),
        "min_speed": segment["Speed"].min(),
        "max_speed": segment["Speed"].max(),
        "avg_throttle": segment["Throttle"].mean(),
        "brake_ratio": segment["Brake"].mean(),  # true = 1,false = 0
        "segment_length": segment["Distance"].iloc[-1] - segment["Distance"].iloc[0],
        "segment_time": (
            segment["Time"].iloc[-1] - segment["Time"].iloc[0]
        ).total_seconds(),
        "driver": driver
    }

    return features
