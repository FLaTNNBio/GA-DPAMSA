import random
import os

# Parametri di configurazione
num_sequences = 4
sequence_length = 101
mutation_rate = 0.10  # Tasso di mutazione 10%
gap_rate = 0.05  # Tasso di inserimento gap
number_of_dataset = 50
min_score_threshold = 50  # Soglia minima del punteggio di allineamento
DATASET_NAME = 'new_training_dataset1_4x101bp'

# Numero e lunghezze dei blocchi conservati
conserved_block_sizes = [10, 10]  # Lista delle lunghezze dei blocchi conservati

FILE_NAME_SCRIPT_OUTPUT = f'./datasets/{DATASET_NAME}.py'
FASTA_OUTPUT = f'./datasets/fasta_files/{DATASET_NAME}'

if not os.path.exists(FASTA_OUTPUT):
    os.makedirs(FASTA_OUTPUT)


# Genera una sequenza DNA casuale
def generate_random_dna_sequence(BP):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(BP))


# Introduce mutazioni controllate solo al di fuori dei blocchi conservati
def mutate_sequence(sequence, mutation_rate, conserved_blocks, insert_positions):
    nucleotides = ['A', 'T', 'C', 'G']
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        # Controlla se l'indice è all'interno di un blocco conservato
        if any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks)):
            continue  # Salta la mutazione all'interno del blocco conservato
        # Applica mutazione al di fuori del blocco conservato
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(nucleotides)
    return ''.join(mutated_sequence)


# Inserisce gap casuali nella sequenza solo al di fuori dei blocchi conservati
def insert_random_gaps(sequence, gap_rate, conserved_blocks, insert_positions, max_gaps=None):
    gapped_sequence = list(sequence)
    gap_count = 0
    for i in range(len(gapped_sequence)):
        # Controlla se l'indice è all'interno di un blocco conservato
        if any(insert <= i < insert + len(block) for insert, block in zip(insert_positions, conserved_blocks)):
            continue  # Salta l'inserimento di gap all'interno del blocco conservato
        # Inserisce gap solo al di fuori dei blocchi conservati
        if random.random() < gap_rate:
            gapped_sequence[i] = '-'
            gap_count += 1
            if max_gaps and gap_count >= max_gaps:
                break
    return ''.join(gapped_sequence)


# Calcola il punteggio di allineamento per una coppia di sequenze
def calculate_alignment_score(sequences):
    total_score = 0
    for k in range(len(sequences[0])):  # Itera sulle posizioni
        for i in range(len(sequences) - 1):  # Itera sugli indici delle sequenze fino a len(sequences) - 1
            for j in range(i + 1, len(sequences)):  # Itera sugli indici successivi a i
                x = sequences[i][k]
                y = sequences[j][k]
                if x == '-' or y == '-':
                    total_score += -4
                elif x == y:
                    total_score += 4
                else:
                    total_score += -4
    return total_score


# Genera dataset con blocchi comuni fino a raggiungere la soglia di allineamento
def generate_dataset_with_common_blocks(seq_length, conserved_blocks, mutation_rate, gap_rate, min_score_threshold):
    while True:
        # Genera i blocchi conservati per il dataset
        conserved_blocks_sequences = [generate_random_dna_sequence(length) for length in conserved_blocks]

        # Genera le posizioni casuali dei blocchi conservati per le sequenze
        insert_positions = [random.randint(0, seq_length - len(block)) for block in conserved_blocks_sequences]

        # Genera le sequenze con blocchi conservati inseriti nelle stesse posizioni
        sequences = []
        for _ in range(num_sequences):
            # Crea una regione variabile
            variable_region = generate_random_dna_sequence(seq_length)

            # Inserisce ogni blocco conservato alla posizione specifica
            for i, block in enumerate(conserved_blocks_sequences):
                position = insert_positions[i]
                variable_region = (
                        variable_region[:position] +
                        block +
                        variable_region[position + len(block):]
                )

            # Introduci mutazioni e gap solo al di fuori dei blocchi conservati
            mutated_sequence = mutate_sequence(variable_region, mutation_rate, conserved_blocks_sequences,
                                               insert_positions)
            gapped_sequence = insert_random_gaps(mutated_sequence, gap_rate, conserved_blocks_sequences,
                                                 insert_positions)
            sequences.append(gapped_sequence)

        # Calcola il punteggio di allineamento
        score = calculate_alignment_score(sequences)
        if score >= min_score_threshold:
            return sequences, conserved_blocks_sequences, insert_positions, score


# Scrive le sequenze in un file FASTA
def write_fasta_file(filename, sequences):
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            header = f">Sequence_{i + 1}\n"
            f.write(header)
            f.write(seq + "\n")


# Colleziona e struttura i dataset in un file Python
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


# Genera dataset finché il punteggio è nel range desiderato
fasta_files = []
for dataset in range(number_of_dataset):
    sequences, conserved_blocks, insert_positions, score = generate_dataset_with_common_blocks(
        sequence_length, conserved_block_sizes, mutation_rate, gap_rate, min_score_threshold
    )

    # Stampa i blocchi conservati e le loro posizioni
    for i, block in enumerate(conserved_blocks):
        print(f"Dataset {dataset}: Blocco conservato {i + 1} '{block}' inserito alla posizione {insert_positions[i]}")

    # Salva il dataset
    fasta_filename = f'test{dataset}.fasta'
    fasta_filename = os.path.join(FASTA_OUTPUT, fasta_filename)
    write_fasta_file(fasta_filename, sequences)
    fasta_files.append(fasta_filename)
    print(f"Fasta file created in {fasta_filename} with score {score}")

write_dataset_dpamsa(fasta_files, FILE_NAME_SCRIPT_OUTPUT)
print(f"Dataset file created in {FILE_NAME_SCRIPT_OUTPUT}")
