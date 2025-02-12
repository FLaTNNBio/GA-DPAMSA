import os

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


DATASET_ROW = 6
DATASET_COLUMN = 30
NUM_TOTAL_RANGES = int((DATASET_ROW / AGENT_WINDOW_ROW) * (DATASET_COLUMN/AGENT_WINDOW_COLUMN))


# ASSERTIONS
assert 0 < BATCH_SIZE <= REPLAY_MEMORY_SIZE, "batch size must be in the range of 0 to the size of replay memory."
assert ALPHA > 0, "alpha must be greater than 0."
assert 0 <= GAMMA <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= EPSILON <= 1, "epsilon must be in the range of 0 to 1."
assert 0 <= DELTA <= EPSILON, "delta must be in the range of 0 to epsilon."
assert 0 < DECREMENT_ITERATION, "decrement iteration must be greater than 0."


# PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_DATASETS_PATH = os.path.join(PROJECT_ROOT, "datasets")
FASTA_FILES_PATH = os.path.join(BASE_DATASETS_PATH, "fasta_files")
TRAINING_DATASET_PATH = os.path.join(BASE_DATASETS_PATH, "training_dataset")
INFERENCE_DATASET_PATH = os.path.join(BASE_DATASETS_PATH, "inference_dataset")

DPAMSA_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "DPAMSA", "weights")

BASE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
REPORTS_PATH = os.path.join(BASE_RESULTS_PATH, "reports")
DPAMSA_REPORTS_PATH = os.path.join(REPORTS_PATH, "DPAMSA")
GA_DPAMSA_REPORTS_PATH = os.path.join(REPORTS_PATH, "GA-DPAMSA")
BENCHMARKS_PATH = os.path.join(BASE_RESULTS_PATH, "benchmarks")
TOOLS_OUTPUT_PATH = os.path.join(BASE_RESULTS_PATH, "tools_output")
CSV_PATH = os.path.join(BENCHMARKS_PATH, "csv")
INFERENCE_CSV_PATH = os.path.join(CSV_PATH, "inference")
DPAMSA_INF_CSV_PATH = os.path.join(INFERENCE_CSV_PATH, "DPAMSA")
GA_DPAMSA_INF_CSV_PATH = os.path.join(INFERENCE_CSV_PATH, "GA-DPAMSA")
CHARTS_PATH = os.path.join(BENCHMARKS_PATH, "charts")

# Ensure directories exist, creating them if they don't
REQUIRED_DIRECTORIES = [
    DPAMSA_WEIGHTS_PATH,
    BASE_RESULTS_PATH,
    DPAMSA_REPORTS_PATH,
    GA_DPAMSA_REPORTS_PATH,
    BENCHMARKS_PATH,
    TOOLS_OUTPUT_PATH,
    CSV_PATH,
    INFERENCE_CSV_PATH,
    DPAMSA_INF_CSV_PATH,
    GA_DPAMSA_INF_CSV_PATH,
    CHARTS_PATH
]
for path in REQUIRED_DIRECTORIES:
    if not os.path.exists(path):
        os.makedirs(path)


# TOOLS
TOOLS = {
    'ClustalOmega': {
        'command': lambda file_path, output_dir: ['clustalo', '-i', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalOmega'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalOmega')
    },
    'MSAProbs': {
        'command': lambda file_path, output_dir: ['msaprobs', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MSAProbs'),
        'report_dir': os.path.join(REPORTS_PATH, 'MSAProbs')
    },
    'ClustalW': {
        'command': lambda file_path, output_dir: ['clustalw', f'-INFILE={file_path}',
                                                  '-OUTPUT=FASTA', f'-OUTFILE={output_dir}'],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalW'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalW')
    },
    'MAFFT': {
        'command': lambda file_path, output_dir: f"mafft --auto {file_path} > {output_dir}",
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MAFFT'),
        'report_dir': os.path.join(REPORTS_PATH, 'MAFFT')
    },
    'MUSCLE5': {
        'command': lambda file_path, output_dir: ['muscle5', '-align', file_path, '-output', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MUSCLE5'),
        'report_dir': os.path.join(REPORTS_PATH, 'MUSCLE5')
    },
    'UPP': {
        'command': lambda file_path, output_dir: ['run_upp.py', '-s', file_path, '-m', 'dna', '-d', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'UPP'),
        'report_dir': os.path.join(REPORTS_PATH, 'UPP')
    },
    'PASTA': {
        'command': lambda file_path, output_dir: ['run_pasta.py', '-i', file_path, '-o', output_dir],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'PASTA'),
        'report_dir': os.path.join(REPORTS_PATH, 'PASTA')
    }
}
