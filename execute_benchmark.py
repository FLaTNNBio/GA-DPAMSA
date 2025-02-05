import subprocess
import os
import utils
import csv

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

# Crea la directory clustalw_output se non esiste
clustalw_output_folder = './clustalw_output'
os.makedirs(clustalw_output_folder, exist_ok=True)

for file in files:
    file_path = dataset_folder + file

    # Comandi per eseguire gli strumenti di allineamento
    command_clustalo = f'clustalo -i {file_path}'
    command_msaprobs = f"/home/musimathicslab/Scrivania/tools_GA-DPAMSA/MSAProbs-0.9.7/MSAProbs/msaprobs {file_path}"
    command_clusaltw = f"clustalw {file_path} -OUTPUT=FASTA -OUTFILE=./clustalw_output/{file}"

    # Esegui Clustal Omega
    alignment_output_clustalo = subprocess.check_output(command_clustalo, shell=True, encoding='utf-8')

    # Esegui MSAProbs
    alignment_output_msaprobs = subprocess.check_output(command_msaprobs, shell=True, encoding='utf-8')

    # Esegui ClustalW e leggi l'output
    subprocess.check_output(command_clusaltw, shell=True, encoding='utf-8')
    alignment_output_clustalw = subprocess.check_output(f'cat ./clustalw_output/{file}', shell=True, encoding='utf-8')

    # Calcola i Sum of Pairs per ogni tool
    sum_of_pairs_clustalo = utils.sum_of_pairs_from_fasta(alignment_output_clustalo)
    sum_of_pairs_msaprobs = utils.sum_of_pairs_from_fasta(alignment_output_msaprobs)
    sum_of_pairs_clustalw = utils.sum_of_pairs_from_fasta(alignment_output_clustalw)

    # Aggiungi i risultati alla lista per il CSV
    csv_data.append([file, sum_of_pairs_clustalo, sum_of_pairs_msaprobs, sum_of_pairs_clustalw])

    # Aggiungi i valori alla lista per il calcolo delle medie
    sp_clustalo_values.append(sum_of_pairs_clustalo)
    sp_msaprobs_values.append(sum_of_pairs_msaprobs)
    sp_clustalw_values.append(sum_of_pairs_clustalw)

# Salva i risultati delle SP per ogni file
with open(f'./result/benchmark/{dataset_name}_clustalo_msaprobs_clustalw.csv', mode='w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    writer.writerow(["File name", "SP_ClustalOmega", "SP_MSAProbs", "SP_ClustalW"])
    writer.writerows(csv_data)

# Calcola le medie delle SP per ogni tool
average_sp_clustalo = sum(sp_clustalo_values) / len(sp_clustalo_values)
average_sp_msaprobs = sum(sp_msaprobs_values) / len(sp_msaprobs_values)
average_sp_clustalw = sum(sp_clustalw_values) / len(sp_clustalw_values)

# Salva i risultati medi in un altro file CSV
with open(f'./result/benchmark/{dataset_name}_average_sp.csv', mode='w', newline='') as avg_csv:
    writer = csv.writer(avg_csv)
    writer.writerow(["Tool", "Average SP"])
    writer.writerow(["ClustalOmega", average_sp_clustalo])
    writer.writerow(["MSAProbs", average_sp_msaprobs])
    writer.writerow(["ClustalW", average_sp_clustalw])

# Rimuovi i file DND generati da Clustal Omega
for file in os.listdir(dataset_folder):
    if file.endswith('.dnd'):
        os.remove(os.path.join(dataset_folder, file))

# Rimuovi la directory clustalw_output e i suoi contenuti
for file in os.listdir(clustalw_output_folder):
    os.remove(os.path.join(clustalw_output_folder, file))
os.rmdir(clustalw_output_folder)