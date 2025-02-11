import random
import os
import config
from tqdm import tqdm

import utils

# Configuration parameters
num_sequences = 3
sequence_length = 30
mutation_rate = 0.10  # Mutation rate 10%
gap_rate = 0.05  # Gap insertion rate
number_of_dataset = 50
min_score_threshold = 10  # Minimum alignment score threshold
DATASET_NAME = 'synthetic_dataset_3x30bp'

# Number and lengths of conserved blocks
conserved_block_sizes = [5]  # List of conserved block lengths

FILE_NAME_SCRIPT_OUTPUT = f'{utils.TRAINING_DATASET_PATH}/{DATASET_NAME}.py'
FASTA_OUTPUT = f'{utils.FASTA_FILES_PATH}/{DATASET_NAME}'

if not os.path.exists(FASTA_OUTPUT):
    os.makedirs(FASTA_OUTPUT)


# Generates a random DNA sequence
def generate_random_dna_sequence(BP):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(BP))


# Introduces controlled mutations only outside conserved blocks
def mutate_sequence(sequence, mutation_rate, conserved_blocks, insert_positions):
    nucleotides = ['A', 'T', 'C', 'G']
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        # Checks if the index is within a conserved block
        if any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks)):
            continue  # Skips mutation within the conserved block
        # Applies mutation outside the conserved block
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(nucleotides)
    return ''.join(mutated_sequence)


# Inserts random gaps into the sequence only outside conserved blocks
def insert_random_gaps(sequence, gap_rate, conserved_blocks, insert_positions, max_gaps=None):
    gapped_sequence = list(sequence)
    gap_count = 0
    for i in range(len(gapped_sequence)):
        # Checks if the index is within a conserved block
        if any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks)):
            continue  # Skips gap insertion within the conserved block
        # Inserts gaps only outside conserved blocks
        if random.random() < gap_rate:
            gapped_sequence[i] = '-'
            gap_count += 1
            if max_gaps and gap_count >= max_gaps:
                break
    return ''.join(gapped_sequence)


# Calculates the alignment score for a pair of sequences
def calculate_alignment_score(sequences):
    total_score = 0
    for k in range(len(sequences[0])):  # Iterates over positions
        for i in range(len(sequences) - 1):  # Iterates over sequence indices up to len(sequences) - 1
            for j in range(i + 1, len(sequences)):  # Iterates over indices after i
                x = sequences[i][k]
                y = sequences[j][k]
                if x == '-' or y == '-':
                    total_score += -4
                elif x == y:
                    total_score += 4
                else:
                    total_score += -4
    return total_score


# Generates a dataset with common blocks until the alignment threshold is reached
def generate_dataset_with_common_blocks(seq_length, conserved_blocks, mutation_rate, gap_rate, min_score_threshold):
    while True:
        # Generates the conserved blocks for the dataset
        conserved_blocks_sequences = [generate_random_dna_sequence(length) for length in conserved_blocks]

        # Generates random positions for the conserved blocks in the sequences
        insert_positions = [random.randint(0, seq_length - len(block)) for block in conserved_blocks_sequences]

        # Generates sequences with conserved blocks inserted at the same positions
        sequences = []
        for _ in range(num_sequences):
            # Creates a variable region
            variable_region = generate_random_dna_sequence(seq_length)

            # Inserts each conserved block at the specified position
            for i, block in enumerate(conserved_blocks_sequences):
                position = insert_positions[i]
                variable_region = (
                        variable_region[:position] +
                        block +
                        variable_region[position + len(block):]
                )

            # Introduces mutations and gaps only outside conserved blocks
            mutated_sequence = mutate_sequence(variable_region, mutation_rate, conserved_blocks_sequences,
                                               insert_positions)
            gapped_sequence = insert_random_gaps(mutated_sequence, gap_rate, conserved_blocks_sequences,
                                                 insert_positions)
            sequences.append(gapped_sequence)

        # Calculates the alignment score
        score = calculate_alignment_score(sequences)
        if score >= min_score_threshold:
            return sequences, conserved_blocks_sequences, insert_positions, score


# Writes the sequences to a FASTA file
def write_fasta_file(filename, sequences):
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            header = f">Sequence_{i + 1}\n"
            f.write(header)
            f.write(seq + "\n")


# Collects and structures datasets in a Python file
def write_dataset_dpamsa(fasta_files, output_file):
    sequences = {}
    for file in fasta_files:
        sequences[file] = {}
        with open(file, 'r') as f:
            lines = f.readlines()
            seq = ''
            seq_name = ''
            for line in lines:
                if line.startswith('>'):
                    if seq_name != '':
                        sequences[file][seq_name] = seq
                    seq_name = line.strip().lstrip('>')
                    seq = ''
                else:
                    seq += line.strip()
            if seq_name != '':
                sequences[file][seq_name] = seq
    fasta_file_names = []
    for file in fasta_files:
        filename = os.path.basename(file).split('.')[0]
        fasta_file_names.append(filename)

    file_content = f"""
file_name = '{os.path.basename(output_file)}'

datasets = {fasta_file_names}
    """

    for i, filename in enumerate(fasta_files):
        dataset_name = f"dataset{i}"
        if filename in sequences:
            dataset_sequences = sequences[filename]
            file_content += f"\n{fasta_file_names[i]} = "
            sequence_list = list(dataset_sequences.values())
            file_content += f'{sequence_list}\n'

    with open(output_file, 'w') as f:
        f.write(file_content)


if __name__ == "__main__":
    fasta_files = []
    for dataset in tqdm(range(number_of_dataset)):
        sequences, conserved_blocks, insert_positions, score = generate_dataset_with_common_blocks(
            sequence_length, conserved_block_sizes, mutation_rate, gap_rate, min_score_threshold
        )

        # Stampa i dettagli dei blocchi conservati
        for i, block in enumerate(conserved_blocks):
            print(f"Dataset {dataset}: Conserved block {i + 1} '{block}' inserted at position {insert_positions[i]}")

        # Salva il dataset in formato FASTA
        fasta_filename = f'test{dataset}.fasta'
        fasta_filename = os.path.join(FASTA_OUTPUT, fasta_filename)
        write_fasta_file(fasta_filename, sequences)
        fasta_files.append(fasta_filename)
        print(f"Fasta file created in {fasta_filename} with score {score}")

    # Crea il file Python strutturato per i dataset
    write_dataset_dpamsa(fasta_files, FILE_NAME_SCRIPT_OUTPUT)
    print(f"Dataset file created in {FILE_NAME_SCRIPT_OUTPUT}")