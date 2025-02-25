import datasets.inference_dataset.dataset1_3x30bp as inference_dataset
import utils
from DPAMSA.env import Environment
from GA import GA
import config
import csv
import os
from tqdm import tqdm

GA_MODE = 'mo'
DATASET = inference_dataset
INFERENCE_MODEL = 'model_3x30'
DEBUG_MODE = False


def output_parameters():
    print("---------- DPAMSA parameters -----------------")
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.MAX_EPISODE))
    print("Batch size: {}".format(config.BATCH_SIZE))
    print("Replay memory size: {}".format(config.REPLAY_MEMORY_SIZE))
    print("Alpha: {}".format(config.ALPHA))
    print("Epsilon: {}".format(config.EPSILON))
    print("Gamma: {}".format(config.GAMMA))
    print("Delta: {}".format(config.DELTA))
    print("Decrement iteration: {}".format(config.DECREMENT_ITERATION))
    print("Update iteration: {}".format(config.UPDATE_ITERATION))
    print("Device: {}".format(config.DEVICE_NAME))
    print("-------- Genetic Algorithm parameters ---------")
    print(f"Window size:{config.AGENT_WINDOW_ROW}x{config.AGENT_WINDOW_COLUMN}")
    print(f"Population Size: {config.POPULATION_SIZE}")
    print(f"Number of iteration: {config.GA_ITERATIONS}")
    print(f"Selection Rate: {config.SELECTION_RATE * 100}%")
    print(f"Mutation Rate: {config.MUTATION_RATE * 100}%")
    print('\n')


def inference(mode, dataset=DATASET, start=0, end=-1, model_path='model_3x30', debug=False, truncate_file=True):
    """
        Run the genetic algorithm with a specific inference mode.

        Parameters:
        -----------
        - mode (str): The mode of operation. Must be one of:
            * 'sp'  -> Sum of Pairs mode
            * 'cs'  -> Column Score mode
            * 'mo'  -> Multi-Objective mode
        - dataset: The dataset containing sequences.
        - start (int): Starting index of datasets to process.
        - end (int): Ending index of datasets to process (default: -1 for all).
        - model_path (str): Path to the model used for mutation.
        - debug (bool): Whether to run GA in debug mode (Detailed Real-Time vision of the algorithm operating) or not.
        - truncate_file (bool): Whether to overwrite report and CSV files.

        Raises:
        -------
        - Exception: If an invalid mode is provided.
        """
    # Mode validation
    valid_modes = {'sp', 'cs', 'mo'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose one of {valid_modes}.")

    # Paths definition
    tag = os.path.splitext(dataset.file_name)[0]
    mode_tag = {"sp": "Max_SP", "cs": "Max_CS", "mo": "MO"}[mode]
    report_file_name = os.path.join(config.GA_DPAMSA_REPORTS_PATH, f"{tag}_{mode_tag}.txt")
    csv_file_name = os.path.join(config.GA_DPAMSA_INF_CSV_PATH, f"{tag}_{mode_tag}_GA_DPAMSA_results.csv")

    # Create or truncate results files
    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                             "Exact Matches (EM)", "Column Score (CS)"])

    # Show DPAMSA and GA configs
    output_parameters()

    # Inference loop
    datasets_to_process = dataset.datasets[start:end if end != -1 else len(dataset.datasets)]
    for index, dataset_name in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):

        if not hasattr(dataset, dataset_name):
            continue
        seqs = getattr(dataset, dataset_name)

        env = Environment(seqs, convert_data=False)

        ga = GA(seqs, mode)
        best_alignment = ga.run(model_path, debug)
        Environment.set_alignment(env, best_alignment)

        # Compute metrics
        metrics = utils.calculate_metrics(env)

        # Create report
        report = (
            f"File: {dataset_name}\n"
            f"Number of Sequences (QTY): {metrics['QTY']}\n"
            f"Alignment Length (AL): {metrics['AL']}\n"
            f"Sum of Pairs (SP): {metrics['SP']}\n"
            f"Exact Matches (EM): {metrics['EM']}\n"
            f"Column Score (CS): {metrics['CS']:.3f}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        # Save results to files
        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([dataset_name, metrics['QTY'], metrics['AL'], metrics['SP'], metrics['EM'], metrics['CS']])

    print(f"\nInference completed successfully.")
    print(f"Report saved at: {report_file_name}")
    print(f"CSV saved at: {csv_file_name}\n\n")


if __name__ == "__main__":
    """
       Available inference modes:
       - 'sp'  -> Sum of Pairs mode
       - 'cs'  -> Column Score mode
       - 'mo'  -> Multi-Objective mode
    """
    inference(mode=GA_MODE, dataset=DATASET, model_path=INFERENCE_MODEL, debug=DEBUG_MODE)
