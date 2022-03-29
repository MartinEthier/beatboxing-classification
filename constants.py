# Class IDs, use indices to map to class number and back
CLASS_MAP = ['sd', 'kd', 'hho', 'hhc']

# Based on plots generated in preprocess script
MAX_TIME = 0.95 # in seconds
SAMPLE_RATE = 44100
MAX_LENGTH = int(MAX_TIME * SAMPLE_RATE)

# Went through all audio files and found the ones that had silent regions at the end
FILE_TRIMS = [
    # (subset, participant, file type, timestamp)
    ("Personal", "24", "Snare", 11.0),
    ("Personal", "24", "Kick", 11.1),
    ("Personal", "24", "HHopened", 12.12),
    ("Personal", "24", "HHclosed", 10.92),
    ("Personal", "20", "HHopened", 12.5),
    ("Personal", "20", "HHclosed", 12.35),
    ("Fixed", "26", "Snare", 14.6),
    ("Fixed", "26", "Kick", 14.44),
    ("Fixed", "26", "HHclosed", 15.57),
    ("Fixed", "26", "Improvisation", 14.3),
    ("Fixed", "13", "HHclosed", 9.9),
    ("Fixed", "8", "HHopened", 9.9),
    ("Fixed", "8", "HHclosed", 9.69),
    ("Fixed", "6", "Snare", 10.96),
    ("Fixed", "6", "Kick", 11.0),
    ("Fixed", "6", "HHopened", 11.04),
    ("Fixed", "6", "HHclosed", 11.04),
    ("Fixed", "20", "HHopened", 12.85),
    ("Fixed", "20", "HHclosed", 12.47),
    ("Fixed", "7", "Snare", 10.99),
    ("Fixed", "7", "Kick", 10.99),
    ("Fixed", "7", "HHopened", 11.0),
    ("Fixed", "7", "HHclosed", 11.0)
]
