import config
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
import subprocess
from tqdm import tqdm
from DPAMSA.env import Environment
from config import CSV_PATH, INFERENCE_CSV_PATH, CHARTS_PATH, TOOLS, GA_DPAMSA_INF_CSV_PATH, DPAMSA_INF_CSV_PATH
from mainGA import inference as ga_inference
from DPAMSA.main import inference as dpamsa_inference


# --------
# GA UTILS
# --------
def is_overlap(range1, range2):
    from_row1, to_row1, from_column1, to_column1 = range1
    from_row2, to_row2, from_column2, to_column2 = range2

    # Check overlap
    overlap_row = from_row1 < to_row2 and to_row1 > from_row2
    overlap_column = from_column1 < to_column2 and to_column1 > from_column2

    return overlap_row and overlap_column


#TODO: can generate error in case the sub-board is not a multiple of the main-board in terms of number of row and column
#Possibile fix: controllare to_column con il valore effettivo del numero di colonne, se to_column >, allora to_column diventa pari al numero di colonne totali della board
def get_sum_of_pairs(chromosome,from_row,to_row,from_column,to_column):
    score = 0
    for i in range(from_column, to_column):
        for j in range(from_row,to_row):
            for k in range(j + 1, to_row):
                if chromosome[j][i] == 5 or chromosome[k][i] == 5:
                    score += config.GAP_PENALTY
                elif chromosome[j][i] == chromosome[k][i]:
                    score += config.MATCH_REWARD
                elif chromosome[j][i] != chromosome[k][i]:
                    score += config.MISMATCH_PENALTY
    return score


def check_overlap(new_range,used_ranges):
    for existing_range in used_ranges:
        if is_overlap(new_range, existing_range):
            return True  # range overlap
    return False  # no overlap


def get_all_different_sub_range(individual,m_prime,n_prime):
    m = len(individual)
    #give as n the minimum length of a sequence, this in case we have to align sequence of different size
    n = min(len(genes) for genes in individual)

    m_prime = config.AGENT_WINDOW_ROW
    n_prime = config.AGENT_WINDOW_COLUMN

    #Calculate all the possible sub-board of size m_prime, n_prime from the main board
    unique_ranges = []
    for i in range(m):
        for j in range(n):
            from_row = i 
            to_row = i + m_prime
            from_column = j
            to_column = j + n_prime
            #Check if the range is a range already covered by another interval (we want all different sub-board)
            if(check_overlap((from_row,to_row,from_column,to_column),unique_ranges) == False):
                                #we want only equal range, if we create gaps and the board is if we create gap and the board is not partitionable, we only take the largest number of sub-boards of the same size as the board on which the algorithm was trained
                        if not (to_row > m or to_column > n):    
                            unique_ranges.append((from_row,to_row,from_column,to_column))
    
    return unique_ranges


def calculate_worst_fitted_sub_board(individual):
    #Get all the possible sub-board from the individual (the main board)
    unique_ranges = get_all_different_sub_range(individual,config.AGENT_WINDOW_ROW,config.AGENT_WINDOW_COLUMN)
    sub_board_score = []

    #For every sub-board, calculate che sum of pair
    for from_row,to_row,from_column,to_column in unique_ranges:
        score = get_sum_of_pairs(individual,from_row,to_row, from_column, to_column)
        sub_board_score.append(((score),(from_row,to_row,from_column,to_column)))
    
    #Find the sub-board with the worst sum-of-pairs score
    worst_score_subboard = min(sub_board_score, key=lambda x: x[0])

    return worst_score_subboard


#Function for generate a list of num_random_el elements of different random number
def casual_number_generation(start_range, final_range, num_random_el):
    generated_number = set()
    while len(generated_number) < num_random_el:
        num = random.randint(start_range, final_range)
        generated_number.add(num)
    
    return list(generated_number)


#Return the first num_individuals individuals with the worst score
def get_index_of_the_worst_fitted_individuals(population_sorted,num_individuals):
    #Sort the population based on the score
    population_score_sorted = sorted(population_sorted, key=lambda x: x[1])
    #Get the index of the worst fitted individuals
    worst_fitted_individual = [item[0] for item in population_score_sorted[:num_individuals]]
    
    return worst_fitted_individual


#Return the first num_individuals individuals with the worst score
def get_index_of_the_best_fitted_individuals(population_sorted,num_individuals):
    #Sort the population based on the score
    population_score_sorted = sorted(population_sorted, key=lambda x: x[1],reverse=True)
    #Get the index of the best fitted individuals
    best_fitted_individual = [item[0] for item in population_score_sorted[:num_individuals]]
    
    return best_fitted_individual


def check_if_there_are_all_gaps(row,from_index):
    for i in range(from_index, len(row)):
        if row[i] != 5:
            return False    
    return from_index - 1


def clean_unnecessary_gaps(aligned_sequence):
    indexes_to_start_clean = []
    for index_el,row in enumerate(aligned_sequence):
        for index_col,el in enumerate(row):
            if el == 5:
                result = check_if_there_are_all_gaps(row,index_col + 1)
                if result != False:
                    indexes_to_start_clean.append(result)
                    break
    try:
        index_to_start = max(indexes_to_start_clean)
        for row in aligned_sequence:
            del row[index_to_start:len(row)]
    except:
        return
    ''' This is util for the mainGA to get the sub board in the correct way and also for replace after the mutation
    row_genes = individual[from_row:to_row]
    for genes in row_genes:
        sub_genes = genes[from_column:to_column]      
        sub_board.append(sub_genes)

    all_sub_board.append(sub_board)
    '''    


# ------------------
# BENCHMARKING UTILS
# ------------------
def calculate_metrics(env):

    alignment_length = len(env.aligned[0])
    num_sequences = len(env.aligned)
    sp_score = env.calc_score()
    exact_matches = env.calc_exact_matched()
    column_score = exact_matches / alignment_length

    return {
        "AL": alignment_length,
        "QTY": num_sequences,
        "SP": sp_score,
        "EM": exact_matches,
        "CS": column_score
    }


def parse_fasta_to_sequences(fasta_content):
    sequences = []
    current_sequence = []

    for line in fasta_content.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_sequence:
                sequences.append(''.join(current_sequence))
                current_sequence = []
        else:
            current_sequence.append(line)

    # Aggiungi l'ultima sequenza se presente
    if current_sequence:
        sequences.append(''.join(current_sequence))

    return sequences


def display_menu():

    print("Seleziona l'opzione per il benchmarking:")
    print("1. GA-DPAMSA vs DPAMSA")
    print("2. GA-DPAMSA vs Tools")
    print("3. GA-DPAMSA vs DPAMSA vs Tools")

    while True:
        try:
            choice = int(input("Inserisci il numero della tua scelta (1, 2, 3): "))
            if choice in [1, 2, 3]:
                return choice
            else:
                print("Seleziona un'opzione valida (1, 2, 3).")
        except ValueError:
            print("Input non valido. Inserisci un numero.")


def run_tool_and_generate_report(tool_name, file_paths, dataset_name):
    tool_info = TOOLS[tool_name]

    os.makedirs(tool_info['output_dir'], exist_ok=True)
    # Creazione delle directory per il report e per l'output specifico del dataset
    dataset_output_dir = os.path.join(tool_info['output_dir'], dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    os.makedirs(tool_info['report_dir'], exist_ok=True)

    report_file = os.path.join(tool_info['report_dir'], f"{dataset_name}.txt")
    csv_results = []

    with open(report_file, 'w') as report:
        for file_path in tqdm(file_paths, desc=f"Processing {tool_name}", leave=False):
            file_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(file_name)[0]
            command = tool_info['command'](file_path, os.path.join(dataset_output_dir, file_name))

            # Esecuzione del comando
            if tool_name == 'MAFFT':
                subprocess.run(command, shell=True, stderr=subprocess.DEVNULL, text=True)
            else:
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

            # Gestione specifica dell'output path per UPP e PASTA
            if tool_name == 'UPP':
                output_path = os.path.join(dataset_output_dir, file_name, "output_alignment.fasta")
            elif tool_name == 'PASTA':
                output_path = os.path.join(dataset_output_dir, file_name, f"pastajob.marker001.{file_name_no_ext}.aln")
            else:
                output_path = os.path.join(dataset_output_dir, file_name)

            # Lettura dell'output (da file o da stdout)
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    fasta_content = f.read()
            else:
                fasta_content = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout

            # Parsing del contenuto FASTA per ottenere le sequenze
            aligned_seqs = parse_fasta_to_sequences(fasta_content)

            # Calcolo delle metriche con Environment
            env = Environment(aligned_seqs, convert_data=False)
            Environment.set_alignment(env, aligned_seqs)

            metrics = calculate_metrics(env)

            # Scrittura del report con tutte le metriche
            report.write(f"File: {file_name}\n")
            report.write(f"Alignment Length (AL): {metrics['AL']}\n")
            report.write(f"Number of Sequences (QTY): {metrics['QTY']}\n")
            report.write(f"Sum of Pairs (SP): {metrics['SP']}\n")
            report.write(f"Exact Matches (EM): {metrics['EM']}\n")
            report.write(f"Column Score (CS): {metrics['CS']:.3f}\n")
            report.write(f"Alignment:\n{env.get_alignment()}\n\n")

            # Aggiunta dei risultati per il CSV (con tutte le metriche)
            csv_results.append([
                file_name, metrics['AL'], metrics['QTY'],
                metrics['SP'], metrics['EM'], metrics['CS']
            ])

            # Rimozione dei file .dnd generati da ClustalW
            if tool_name == 'ClustalW':
                dnd_files = glob.glob(os.path.join(os.path.dirname(file_path), '*.dnd'))
                for dnd_file in dnd_files:
                    os.remove(dnd_file)

    return csv_results


def save_inference_csv(csv_data, tool_name, dataset_name):
    # Percorso dove salvare il CSV per il tool specifico
    tool_csv_dir = os.path.join(config.CSV_PATH, tool_name)
    os.makedirs(tool_csv_dir, exist_ok=True)

    # Nome del file CSV
    csv_file_path = os.path.join(tool_csv_dir, f"{dataset_name}_{tool_name}_results.csv")

    # Se csv_data è una lista di liste (come generato da run_tool_and_generate_report)
    if isinstance(csv_data, list):
        columns = ["File Name", "Alignment Length (AL)",
                   "Number of Sequences (QTY)", "Sum of Pairs (SP)",
                   "Exact Matches (EM)", "Column Score (CS)"]
        df = pd.DataFrame(csv_data, columns=columns)
    else:
        # Se è già un DataFrame
        df = pd.read_csv(csv_data)

    df.to_csv(csv_file_path, index=False)
    return csv_file_path  # Restituisce il path per aggiungerlo al dizionario


def run_ga_dpamsa_inference(dataset, dataset_name, model_path):

    ga_inference(dataset=dataset, model_path=model_path, truncate_file=True)
    return os.path.join(GA_DPAMSA_INF_CSV_PATH, f"{dataset_name}_GA_DPAMSA_results.csv")


def run_dpamsa_inference(dataset, dataset_name, model_path):

    dpamsa_inference(dataset=dataset, model_path=model_path, truncate_file=True)
    return os.path.join(DPAMSA_INF_CSV_PATH, f"{dataset_name}_DPAMSA_results.csv")


# ------------------
# DATA VIZ UTILS
# ------------------
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_metrics(tool_csv_paths, dataset_name):
    sum_of_pairs_data = []
    column_score_data = []
    mean_sp = {}
    mean_cs = {}

    # Colori: rosso per GA-DPAMSA, azzurro per gli altri
    color_map = {'GA-DPAMSA': 'red'}

    # Creazione della directory per i grafici del dataset
    dataset_charts_dir = os.path.join(config.CHARTS_PATH, dataset_name)
    os.makedirs(dataset_charts_dir, exist_ok=True)

    # Preparazione dei dati
    for tool, csv_path in tool_csv_paths.items():
        df = pd.read_csv(csv_path)

        # Colori specifici
        color = 'red' if tool == 'GA-DPAMSA' else 'cyan'
        color_map[tool] = color

        # Dati per il box plot
        sum_of_pairs_data.append((tool, df['Sum of Pairs (SP)']))
        column_score_data.append((tool, df['Column Score (CS)']))

        # Dati per il bar plot (valore medio)
        mean_sp[tool] = df['Sum of Pairs (SP)'].mean()
        mean_cs[tool] = df['Column Score (CS)'].mean()

    tools = list(tool_csv_paths.keys())

    # === BOX PLOT: Sum of Pairs (SP) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)  # Linee di sfondo

    boxplot = plt.boxplot(
        [data for _, data in sum_of_pairs_data],
        labels=tools,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        zorder=3  # Sovrapposizione sopra le linee di sfondo
    )

    # Colora i box
    for patch, (tool, _) in zip(boxplot['boxes'], sum_of_pairs_data):
        patch.set_facecolor(color_map[tool])

    # Modifiche estetiche
    plt.title(f'SP Distribution results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Sum of Pairs (SP)', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'sum_of_pairs_distribution.png'), dpi=300)
    plt.close()

    # === BOX PLOT: Column Score (CS) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    boxplot = plt.boxplot(
        [data for _, data in column_score_data],
        labels=tools,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        zorder=3
    )

    for patch, (tool, _) in zip(boxplot['boxes'], column_score_data):
        patch.set_facecolor(color_map[tool])

    plt.title(f'CS Distribution results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Column Score (CS)', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'column_score_distribution.png'), dpi=300)
    plt.close()

    # === BAR PLOT: Mean Sum of Pairs (SP) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    bars = plt.bar(
        mean_sp.keys(),
        mean_sp.values(),
        color=[color_map[tool] for tool in mean_sp.keys()],
        edgecolor='black', linewidth=2,  # Bordo più spesso
        zorder=3
    )

    # Aggiunta dei valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.01 * height),  # Posiziona leggermente sopra la barra
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )

    plt.title(f'Mean SP results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Mean SP', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'mean_sum_of_pairs.png'), dpi=300)
    plt.close()

    # === BAR PLOT: Mean Column Score (CS) ===
    plt.figure(figsize=(12, 8))
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    bars = plt.bar(
        mean_cs.keys(),
        mean_cs.values(),
        color=[color_map[tool] for tool in mean_cs.keys()],
        edgecolor='black', linewidth=2,
        zorder=3
    )

    # Aggiunta dei valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.01 * height),
            f'{height:.3f}',  # Più decimali per il CS
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )

    plt.title(f'Mean CS results for {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Mean CS', fontweight='bold', fontsize=12)
    plt.xticks(fontweight='bold', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(dataset_charts_dir, f'mean_column_score.png'), dpi=300)
    plt.close()
