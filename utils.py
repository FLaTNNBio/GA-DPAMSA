import config
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import subprocess
from DPAMSA.env import Environment
from mainGA import inference as ga_inference
from DPAMSA.main import inference as dpamsa_inference


# ---------
# CONSTANTS
# ---------

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
    CHARTS_PATH
]
for path in REQUIRED_DIRECTORIES:
    if not os.path.exists(path):
        os.makedirs(path)


# TOOLS
TOOLS = {
    'ClustalOmega': {
        'command': lambda file_path: ['clustalo', '-i', file_path],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalOmega'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalOmega')
    },
    'MSAProbs': {
        'command': lambda file_path: ['msaprobs', file_path],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MSAProbs'),
        'report_dir': os.path.join(REPORTS_PATH, 'MSAProbs')
    },
    'ClustalW': {
        'command': lambda file_path: ['clustalw', file_path, '-OUTPUT=FASTA',
                                      f'-OUTFILE={os.path.join(TOOLS_OUTPUT_PATH, "ClustalW", os.path.basename(file_path))}'],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'ClustalW'),
        'report_dir': os.path.join(REPORTS_PATH, 'ClustalW')
    },
    'MAFFT': {
        'command': lambda file_path: ['mafft', '--auto', file_path],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MAFFT'),
        'report_dir': os.path.join(REPORTS_PATH, 'MAFFT')
    },
    'MUSCLE5': {
        'command': lambda file_path: ['muscle5', '-align', file_path, '-output',
                                      os.path.join(TOOLS_OUTPUT_PATH, 'MUSCLE5', os.path.basename(file_path))],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'MUSCLE5'),
        'report_dir': os.path.join(REPORTS_PATH, 'MUSCLE5')
    },
    'UPP': {
        'command': lambda file_path: ['run_upp.py', '-s', file_path, '-m', 'dna', '-d',
                                      os.path.join(TOOLS_OUTPUT_PATH, 'UPP', f"{os.path.basename(file_path)}_output")],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'UPP'),
        'report_dir': os.path.join(REPORTS_PATH, 'UPP')
    },
    'PASTA': {
        'command': lambda file_path: ['run_pasta.py', '-i', file_path, '-o',
                                      os.path.join(TOOLS_OUTPUT_PATH, 'PASTA', f"{os.path.basename(file_path)}_output")],
        'output_dir': os.path.join(TOOLS_OUTPUT_PATH, 'PASTA'),
        'report_dir': os.path.join(REPORTS_PATH, 'PASTA')
    }
}


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


def run_tool_and_generate_report(tool_name, file_paths, dataset_name):
    tool_info = TOOLS[tool_name]

    os.makedirs(tool_info['output_dir'], exist_ok=True)
    os.makedirs(tool_info['report_dir'], exist_ok=True)

    report_file = os.path.join(tool_info['report_dir'], f"{dataset_name}_report.txt")
    csv_results = []

    with open(report_file, 'w') as report:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            command = tool_info['command'](file_path)
            output_path = os.path.join(tool_info['output_dir'], file_name)

            # Esecuzione del comando
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

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
            report.write(f"SP Score: {metrics['SP']}\n")
            report.write(f"Exact Matches (EM): {metrics['EM']}\n")
            report.write(f"Column Score (CS): {metrics['CS']:.3f}\n\n")

            # Aggiunta dei risultati per il CSV (con tutte le metriche)
            csv_results.append([
                file_name, tool_name, metrics['AL'], metrics['QTY'],
                metrics['SP'], metrics['EM'], metrics['CS']
            ])

    return csv_results


def append_results_to_csv(csv_data, csv_filename='tools_metrics.csv'):

    csv_path = os.path.join(CSV_PATH, csv_filename)

    headers = [
        "File Name", "Tool", "Alignment Length (AL)",
        "Number of Sequences (QTY)", "Sum of Pairs (SP)",
        "Exact Matches (EM)", "Column Score (CS)"
    ]

    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        new_data = pd.DataFrame(csv_data, columns=headers)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        combined_data = pd.DataFrame(csv_data, columns=headers)

    combined_data.to_csv(csv_path, index=False)


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


def run_ga_dpamsa_inference(dataset, dataset_name, model_path):

    ga_inference(dataset=dataset, model_path=model_path, truncate_file=True)
    return os.path.join(INFERENCE_CSV_PATH, "GA-DPAMSA",f"{dataset_name}.csv")


def run_dpamsa_inference(dataset, dataset_name, model_path):

    dpamsa_inference(dataset=dataset, model_path=model_path, truncate_file=True)
    return os.path.join(INFERENCE_CSV_PATH, "DPAMSA", f"{dataset_name}_dpamsa.csv")


def aggregate_csvs(main_csv_path, additional_csv_paths):

    # Carica il CSV principale
    main_data = pd.read_csv(main_csv_path)

    # Aggiungi i CSV aggiuntivi
    for csv_path in additional_csv_paths:
        additional_data = pd.read_csv(csv_path)

        # Uniforma le colonne se necessario
        if 'SP Score' in additional_data.columns:
            additional_data.rename(columns={'SP Score': 'SP'}, inplace=True)

        # Aggiungi le colonne mancanti se necessario
        if 'Tool' not in additional_data.columns:
            tool_name = 'GA-DPAMSA' if 'GA-DPAMSA' in csv_path else 'DPAMSA'
            additional_data['Tool'] = tool_name

        # Unisci i dati
        main_data = pd.concat([main_data, additional_data], ignore_index=True)

    # Salva il CSV combinato
    main_data.to_csv(main_csv_path, index=False)


# ------------------
# DATA VIZ UTILS
# ------------------
def plot_metrics(csv_path, dataset_name):

    data = pd.read_csv(csv_path)
    sns.set(style="whitegrid")

    # Definisci i colori: Rosso per GA-DPAMSA, Azzurro per gli altri
    palette = ['#FF4C4C' if tool == 'GA-DPAMSA' else '#5BC0DE' for tool in data['Tool'].unique()]

    ### BoxPlot per SP Score ###
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Tool', y='SP', data=data, palette=palette)
    plt.title(f'Distribuzione degli SP Scores per ogni Tool - {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, f'{dataset_name}_sp_score_boxplot.png'))
    plt.show()

    ### BarPlot per le medie delle metriche ###
    mean_metrics = data.groupby('Tool').mean().reset_index()
    metrics_to_plot = ['SP', 'EM', 'CS']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Tool', y=metric, data=mean_metrics, palette=palette)
        plt.title(f'Valori Medi di {metric} per ogni Tool')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_PATH, f'{dataset_name}_{metric.lower()}_barplot.png'))
        plt.show()
