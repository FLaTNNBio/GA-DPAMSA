import os
import platform
import torch
import math

GAP_PENALTY = -4
MISMATCH_PENALTY = -4
MATCH_REWARD = 4
GA_POPULATION_SIZE = 8
GA_NUM_ITERATION = 3
GA_NUM_MOST_FIT_FOR_ITER = 2
GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER = 0.20 #20%

#This depend from the training dataset given to the DQN
AGENT_WINDOW_ROW = 3
AGENT_WINDOW_COLUMN = 30

DATASET_ROW = 3
DATASET_COLUMN = 30

NUM_TOTAL_RANGES = int((DATASET_ROW / AGENT_WINDOW_ROW) * (DATASET_COLUMN/AGENT_WINDOW_COLUMN))

update_iteration = 128

batch_size = 128
max_episode = 6000
replay_memory_size = 1000

alpha = 0.0001
gamma = 1
epsilon = 0.8
delta = 0.05

decrement_iteration = math.ceil(max_episode * 0.8 / (epsilon // delta))

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'

# Get the absolute path to the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

weight_path_DPAMSA = os.path.join(PROJECT_ROOT, "DPAMSA", "weights")
base_results_dir = os.path.join(PROJECT_ROOT, "results")
report_path_DPAMSA = os.path.join(base_results_dir, "reports", "DPAMSA")
report_path_GA_DPAMSA = os.path.join(base_results_dir, "reports", "GA-DPAMSA")
benchmarks_path = os.path.join(base_results_dir, "benchmarks")
csv_path = os.path.join(benchmarks_path, "csv")
charts_path = os.path.join(benchmarks_path, "charts")


# Ensure directories exist, creating them if they don't
for path in [weight_path_DPAMSA, base_results_dir, report_path_DPAMSA, report_path_GA_DPAMSA, benchmarks_path, csv_path, charts_path]:
    if not os.path.exists(path):
        os.makedirs(path)

assert 0 < batch_size <= replay_memory_size, "batch size must be in the range of 0 to the size of replay memory."
assert alpha > 0, "alpha must be greater than 0."
assert 0 <= gamma <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= epsilon <= 1, "epsilon must be in the range of 0 to 1."
assert 0 <= delta <= epsilon, "delta must be in the range of 0 to epsilon."
assert 0 < decrement_iteration, "decrement iteration must be greater than 0."
