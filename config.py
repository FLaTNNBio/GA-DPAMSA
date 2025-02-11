import torch
import math

# DPAMSA parameters
GAP_PENALTY = -4
MISMATCH_PENALTY = -4
MATCH_REWARD = 4
MAX_EPISODE = 6000
BATCH_SIZE = 128
REPLAY_MEMORY_SIZE = 1000
ALPHA = 0.0001
EPSILON = 0.8
GAMMA = 1
DELTA = 0.05
DECREMENT_ITERATION = math.ceil(MAX_EPISODE * 0.8 / (EPSILON // DELTA))
UPDATE_ITERATION = 128
DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'

# GA parameters
AGENT_WINDOW_ROW = 3
AGENT_WINDOW_COLUMN = 30
GA_POPULATION_SIZE = 8
GA_NUM_ITERATION = 3
GA_NUM_MOST_FIT_FOR_ITER = 2
GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER = 0.20 #20%


DATASET_ROW = 4
DATASET_COLUMN = 101
NUM_TOTAL_RANGES = int((DATASET_ROW / AGENT_WINDOW_ROW) * (DATASET_COLUMN/AGENT_WINDOW_COLUMN))


# ASSERTIONS
assert 0 < BATCH_SIZE <= REPLAY_MEMORY_SIZE, "batch size must be in the range of 0 to the size of replay memory."
assert ALPHA > 0, "alpha must be greater than 0."
assert 0 <= GAMMA <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= EPSILON <= 1, "epsilon must be in the range of 0 to 1."
assert 0 <= DELTA <= EPSILON, "delta must be in the range of 0 to epsilon."
assert 0 < DECREMENT_ITERATION, "decrement iteration must be greater than 0."
