import subprocess
import os
import utils
import shutil
import csv
from tqdm import tqdm

# Nome del dataset e cartella dei file
dataset_name = 'encode_project_dataset_4x101bp'
dataset_folder = f'./datasets/fasta_files/{dataset_name}/'

# Ottieni la lista dei file nella cartella
files = os.listdir(dataset_folder)
files.sort()

# Per salvare i dati delle SP di ogni file
csv_data = []

# Per calcolare le medie delle SP per ogni tool
sp_clustalo_values = []
sp_msaprobs_values = []
sp_clustalw_values = []
sp_tcoffee_values = []
sp_mafft_values = []
sp_muscle5_values = []
sp_pasta_values = []
sp_upp_values = []

# Crea la directory per gli output di altri tool se necessario
output_folders = {
    'clustalw': './clustalw_output',
    'tcoffee': './tcoffee_output',
    'mafft': './mafft_output',
    'muscle5': './muscle5_output',
    'upp': './upp_output'
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Itera sui file con una barra di avanzamento
for file in tqdm(files, desc="Calculating Benchmarks"):
    file_path = os.path.join(dataset_folder, file)

    # Comandi per eseguire gli strumenti di allineamento
    command_clustalo = ['clustalo', '-i', file_path]
    command_msaprobs = ['msaprobs', file_path]
    command_clustalw = ['clustalw', file_path, '-OUTPUT=FASTA', f'-OUTFILE=./clustalw_output/{file}']
    command_tcoffee = ['t_coffee', file_path, '-output=fasta', f'-outfile=./tcoffee_output/{file}']
    command_mafft = ['mafft', '--auto', file_path]
    command_muscle5 = ['muscle5', '-align', file_path, '-output', f'./muscle5_output/{file}']
    command_upp = ['run_upp.py', '-s', file_path, '-m', 'dna', '-d', f'./upp_output/{file}_output']

    # Esegui Clustal Omega
    alignment_output_clustalo = subprocess.run(command_clustalo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout

    # Esegui MSAProbs
    alignment_output_msaprobs = subprocess.run(command_msaprobs, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout

    # Esegui ClustalW
    subprocess.run(command_clustalw, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    with open(f'./clustalw_output/{file}', 'r') as f:
        alignment_output_clustalw = f.read()

    # Esegui T-Coffee
    subprocess.run(command_tcoffee, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    with open(f'./tcoffee_output/{file}', 'r') as f:
        alignment_output_tcoffee = f.read()

    # Esegui MAFFT
    alignment_output_mafft = subprocess.run(command_mafft, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout

    # Esegui MUSCLE5
    subprocess.run(command_muscle5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    with open(f'./muscle5_output/{file}', 'r') as f:
        alignment_output_muscle5 = f.read()

    # Esegui UPP
    subprocess.run(command_upp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    with open(f'./upp_output/{file}_output/output_alignment.fasta', 'r') as f:
        alignment_output_upp = f.read()
    with open(f'./upp_output/{file}_output/output_pasta.fasta', 'r') as f:
        alignment_output_pasta = f.read()

    # Calcola i Sum of Pairs per ogni tool
    sum_of_pairs_clustalo = utils.sum_of_pairs_from_fasta(alignment_output_clustalo)
    sum_of_pairs_msaprobs = utils.sum_of_pairs_from_fasta(alignment_output_msaprobs)
    sum_of_pairs_clustalw = utils.sum_of_pairs_from_fasta(alignment_output_clustalw)
    sum_of_pairs_tcoffee = utils.sum_of_pairs_from_fasta(alignment_output_tcoffee)
    sum_of_pairs_mafft = utils.sum_of_pairs_from_fasta(alignment_output_mafft)
    sum_of_pairs_muscle5 = utils.sum_of_pairs_from_fasta(alignment_output_muscle5)
    sum_of_pairs_pasta = utils.sum_of_pairs_from_fasta(alignment_output_pasta)
    sum_of_pairs_upp = utils.sum_of_pairs_from_fasta(alignment_output_upp)

    # Aggiungi i risultati alla lista per il CSV
    csv_data.append([
        file,
        sum_of_pairs_clustalo,
        sum_of_pairs_msaprobs,
        sum_of_pairs_clustalw,
        sum_of_pairs_tcoffee,
        sum_of_pairs_mafft,
        sum_of_pairs_muscle5,
        sum_of_pairs_pasta,
        sum_of_pairs_upp
    ])

    # Aggiungi i valori alla lista per il calcolo delle medie
    sp_clustalo_values.append(sum_of_pairs_clustalo)
    sp_msaprobs_values.append(sum_of_pairs_msaprobs)
    sp_clustalw_values.append(sum_of_pairs_clustalw)
    sp_tcoffee_values.append(sum_of_pairs_tcoffee)
    sp_mafft_values.append(sum_of_pairs_mafft)
    sp_muscle5_values.append(sum_of_pairs_muscle5)
    sp_pasta_values.append(sum_of_pairs_pasta)
    sp_upp_values.append(sum_of_pairs_upp)

# Salva i risultati delle SP per ogni file
with open(f'./result/benchmark/{dataset_name}_all_tools.csv', mode='w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    writer.writerow([
        "File name",
        "SP_ClustalOmega",
        "SP_MSAProbs",
        "SP_ClustalW",
        "SP_TCoffee",
        "SP_MAFFT",
        "SP_MUSCLE5",
        "SP_PASTA",
        "SP_UPP"
    ])
    writer.writerows(csv_data)

# Calcola le medie delle SP per ogni tool
average_sp_clustalo = sum(sp_clustalo_values) / len(sp_clustalo_values)
average_sp_msaprobs = sum(sp_msaprobs_values) / len(sp_msaprobs_values)
average_sp_clustalw = sum(sp_clustalw_values) / len(sp_clustalw_values)
average_sp_tcoffee = sum(sp_tcoffee_values) / len(sp_tcoffee_values)
average_sp_mafft = sum(sp_mafft_values) / len(sp_mafft_values)
average_sp_muscle5 = sum(sp_muscle5_values) / len(sp_muscle5_values)
average_sp_pasta = sum(sp_pasta_values) / len(sp_pasta_values)
average_sp_upp = sum(sp_upp_values) / len(sp_upp_values)

# Salva i risultati medi in un altro file CSV
with open(f'./result/benchmark/{dataset_name}_average_sp.csv', mode='w', newline='') as avg_csv:
    writer = csv.writer(avg_csv)
    writer.writerow(["Tool", "Average SP"])
    writer.writerow(["ClustalOmega", average_sp_clustalo])
    writer.writerow(["MSAProbs", average_sp_msaprobs])
    writer.writerow(["ClustalW", average_sp_clustalw])
    writer.writerow(["T-Coffee", average_sp_tcoffee])
    writer.writerow(["MAFFT", average_sp_mafft])
    writer.writerow(["MUSCLE5", average_sp_muscle5])
    writer.writerow(["PASTA", average_sp_pasta])
    writer.writerow(["UPP", average_sp_upp])

# Rimuovi i file DND generati da Clustal Omega
for file in os.listdir(dataset_folder):
    if file.endswith('.dnd'):
        os.remove(os.path.join(dataset_folder, file))

# Rimuovi tutti i file .dnd nella root directory del progetto
root_directory = os.getcwd()  # Ottiene la directory corrente (root del progetto)
for file in os.listdir(root_directory):
    if file.endswith('.dnd'):
        file_path = os.path.join(root_directory, file)
        os.remove(file_path)

# Rimuovi le directory di output e i loro contenuti
for folder in output_folders.values():
    if os.path.exists(folder):  # Verifica se la directory esiste
        shutil.rmtree(folder)
