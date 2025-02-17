import csv
import os

import config
from DPAMSA.env import Environment
import datasets.training_dataset.zhang_dataset_3x30 as imported_dataset
from utils import calculate_metrics


def normalize_dataset(dataset, max_length):
    """
    Normalizes sequences in the given dataset, through gap insertion or truncation, to a specified maximum length.
    The modification overwrites the dataset permanently.

    Args:
    -----
    - dataset (module): The imported dataset module.
    - max_length (int): The desired maximum length for the sequences.

    Returns:
    --------
    - None: Modifies the dataset file in-place.
    """

    for index, name in enumerate(dataset.datasets):

        if not hasattr(dataset, name):
            continue
        sequences = getattr(dataset, name)

        modified_sequences = []

        for seq in sequences:
            if len(seq) < max_length:
                modified_seq = seq + "-" * (max_length - len(seq))
            elif len(seq) > max_length:
                modified_seq = seq[:max_length]
            else:
                modified_seq = seq

            modified_sequences.append(modified_seq)

        setattr(dataset, name, modified_sequences)  # Overwrite the dataset in the module

        # Save the modified dataset back to the file, preserving formatting
        with open(dataset.__file__, 'w') as f:
            f.write(f"\nfile_name = '{dataset.file_name}'\n\n")
            f.write(f"datasets = {dataset.datasets}\n\n")  # Write the datasets list with correct formatting
            for dataset_name in dataset.datasets:
                sequences = getattr(dataset, dataset_name)
                f.write(f"{dataset_name} = {sequences}\n\n")  # Write each sub-dataset with correct formatting

    print("\nDataset modification completed and saved to file.")


def calculate_reference_metrics(dataset):
    """
    Calculates baseline metrics for the given unaligned dataset (useful as a reference for the alignments).
    Generates a report file and two csv files (one with the individual scores and one with the average ones).


    Parameters:
    -----------
    - dataset (dict): A dictionary where keys are dataset names (e.g., "dataset_0")
                    and values are lists of sequences (strings).

    Returns:
    --------
    - None: Saves the report and CSV files to disk.

    """
    report_file_name = os.path.join(config.DATASETS_REPORTS_PATH, f"{dataset.file_name}.txt")
    csv_file_name = os.path.join(config.DATASETS_CSV_PATH, f"{dataset.file_name}_scores.csv")
    avg_csv_file_name = os.path.join(config.DATASETS_CSV_PATH, f"{dataset.file_name}_avg_scores.csv")

    # Create or truncate result files
    with open(report_file_name, 'w'):
        pass
    with open(csv_file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                         "Exact Matches (EM)", "Column Score (CS)"])
    total_sp_score = 0
    total_cs_score = 0

    for index, name in enumerate(dataset.datasets):
        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)

        env = Environment(seqs)
        env.set_alignment(seqs)

        metrics = calculate_metrics(env)

        # Crea il report testuale
        report = (
            f"File: {name}\n"
            f"Alignment Length (AL): {metrics['AL']}\n"
            f"Number of Sequences (QTY): {metrics['QTY']}\n"
            f"Sum of Pairs (SP): {metrics['SP']}\n"
            f"Exact Matches (EM): {metrics['EM']}\n"
            f"Column Score (CS): {metrics['CS']:.3f}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        # Scrive il report nel file .rpt
        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        # Scrive i dati nel file CSV
        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, metrics['AL'], metrics['QTY'], metrics['SP'], metrics['EM'], metrics['CS']])

        total_sp_score += metrics['SP']
        total_cs_score += metrics['CS']

    num_datasets = len(dataset.datasets)
    print(num_datasets)
    # Calculate average scores
    avg_sp_score = total_sp_score / num_datasets if num_datasets > 0 else 0
    avg_cs_score = total_cs_score / num_datasets if num_datasets > 0 else 0

    # Save average scores to CSV
    with open(avg_csv_file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Mean SP", "Mean CS"])
        writer.writerow([avg_sp_score, avg_cs_score])

    print(f"\nReport and CSV files generated successfully.")
    print(f"Report saved at: {report_file_name}")
    print(f"Detailed scores saved at: {csv_file_name}")
    print(f"Average scores saved at: {avg_csv_file_name}")


def main():
    """
    Main function to provide a menu-driven interface for dataset manipulation and report generation.
    """

    dataset = imported_dataset  # Assign the imported dataset to a variable

    while True:
        print("\n====== Dataset Manipulation Menu ======")
        print(f"Imported dataset: {dataset.file_name}")
        print("\nOptions:")
        print("1. Apply normalization (pad or truncate sequences)")
        print("2. Calculate baseline metrics")
        print("3. Apply normalization AND calculate baseline metrics")
        print("4. Exit\n")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            max_length = int(input("Enter the desired maximum sequence length: "))
            normalize_dataset(dataset, max_length)

        elif choice == "2":
            calculate_reference_metrics(dataset)

        elif choice == "3":  # New option
            max_length = int(input("Enter the desired maximum sequence length: "))
            normalize_dataset(dataset, max_length)
            calculate_reference_metrics(dataset)

        elif choice == "4":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
