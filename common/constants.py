import numpy as np


ORIGINAL_ACTION_SPACE = [("LEFT", "DOWN"), ("LEFT"), ("LEFT", "UP"), ("DOWN"), (), ("UP"), ("RIGHT", "DOWN"), ("RIGHT"), ("RIGHT", "UP"), ("D"), ("A"), ("W"), ("S"), ("Q"), ("E")]
ACTION_SPACE = [
    ("UP"),
    ("DOWN"),
    ("LEFT"),
    ("RIGHT"),
    ("LEFT", "DOWN"),
    ("LEFT", "UP"),
    ("RIGHT", "DOWN"),
    ("RIGHT", "UP"),
    ()
]
ACTION_TRANSLATION = np.array([ORIGINAL_ACTION_SPACE.index(ACTION_SPACE[i]) for i in range(len(ACTION_SPACE))])
ACTION_MAPPING = {i: ACTION_SPACE[i] for i in range(len(ACTION_SPACE))}
ORIGINAL_ACTION_MAPPING = {i: ORIGINAL_ACTION_SPACE[i] for i in range(len(ORIGINAL_ACTION_SPACE))}
