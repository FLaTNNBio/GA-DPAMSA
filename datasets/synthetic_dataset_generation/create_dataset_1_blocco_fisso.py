import random
import os

# Parametri di configurazione
num_sequences = 6
sequence_length = 30
mutation_rate = 0.10  # Tasso di mutazione 10%
gap_rate = 0.05  # Tasso di inserimento gap
conserved_block_size = 10  # Dimensione del blocco conservato
number_of_dataset = 50
min_score_threshold = 50  # Soglia minima del punteggio di allineamento
max_score_threshold = 200  # Soglia massima del punteggio di allineamento
DATASET_NAME = 'new_training_dataset1_6x30bp'

FILE_NAME_SCRIPT_OUTPUT = f'./datasets/{DATASET_NAME}.py'
FASTA_OUTPUT = f'./datasets/fasta_files/{DATASET_NAME}'

if not os.path.exists(FASTA_OUTPUT):
    os.makedirs(FASTA_OUTPUT)


# Genera una sequenza DNA casuale
def generate_random_dna_sequence(BP):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(BP))


# Introduce mutazioni controllate nella sequenza
def mutate_sequence(sequence, mutation_rate):
    nucleotides = ['A', 'T', 'C', 'G']
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(nucleotides)
    return ''.join(mutated_sequence)


# Inserisce gap casuali nella sequenza
def insert_random_gaps(sequence, gap_rate, max_gaps=None):
    gapped_sequence = list(sequence)
    gap_count = 0
    for i in range(len(gapped_sequence)):
        if random.random() < gap_rate:
            gapped_sequence[i] = '-'
            gap_count += 1
            if max_gaps and gap_count >= max_gaps:
                break
    return ''.join(gapped_sequence)


# Genera una sequenza con un blocco conservato comune per tutte le sequenze
def generate_conserved_variable_sequences(seq_length, conserved_block, mutation_rate, gap_rate):
    sequences = []
    insert_position = random.randint(0, seq_length - len(conserved_block))

    for _ in range(num_sequences):
        # Crea una regione variabile e introduce mutazioni
        variable_region = generate_random_dna_sequence(seq_length)
        mutated_variable = mutate_sequence(variable_region, mutation_rate)

        # Inserisci il blocco conservato alla stessa posizione per tutte le sequenze
        combined_sequence = (
                mutated_variable[:insert_position] +
                conserved_block +
                mutated_variable[insert_position + len(conserved_block):]
        )

        # Inserisci gap casuali
        gapped_sequence = insert_random_gaps(combined_sequence, gap_rate)
        sequences.append(gapped_sequence)

    return sequences, conserved_block, insert_position  # Restituisce anche la posizione di inserimento


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
    while True:
        # Genera un blocco conservato comune per il dataset
        conserved_block = generate_random_dna_sequence(conserved_block_size)

        # Genera sequenze con il blocco conservato
        sequences, conserved_block, insert_position = generate_conserved_variable_sequences(sequence_length,
                                                                                            conserved_block,
                                                                                            mutation_rate, gap_rate)

        # Calcola il punteggio di allineamento
        score = calculate_alignment_score(sequences)
        if min_score_threshold <= score <= max_score_threshold:
            # Stampa la porzione conservata e la posizione
            print(f"Dataset {dataset}: Blocco conservato '{conserved_block}' inserito alla posizione {insert_position}")

            # Se il punteggio è nel range desiderato, salva il dataset
            fasta_filename = f'test{dataset}.fasta'
            fasta_filename = os.path.join(FASTA_OUTPUT, fasta_filename)
            write_fasta_file(fasta_filename, sequences)
            fasta_files.append(fasta_filename)
            print(f"Fasta file created in {fasta_filename} with score {score}")
            break  # Esce dal ciclo while interno per passare al prossimo dataset

write_dataset_dpamsa(fasta_files, FILE_NAME_SCRIPT_OUTPUT)
print(f"Dataset file created in {FILE_NAME_SCRIPT_OUTPUT}")
