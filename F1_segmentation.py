#Detect braking points
def detect_braking_zones(tel, brake_threshold=10):
    print(tel["Brake"].value_counts())
    return tel["Brake"] > brake_threshold

#Find brake ON transitions
def find_brake_starts(tel):
    brake_on = tel["Brake"] == True
    brake_start_indices = tel.index[
        brake_on & (~brake_on.shift(1, fill_value=False))
    ]
    return brake_start_indices



#segments between brake points
def segment_by_braking(tel):
    brake_starts = find_brake_starts(tel)
    segments = []

    for i in range(len(brake_starts)-1):
        start = brake_starts[i]
        end = brake_starts[i+1]
        segment = tel.loc[start:end]
        segments.append(segment)

    return segments

