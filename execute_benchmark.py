import subprocess
import os
import utils
import shutil
import csv
from tqdm import tqdm
from mainGA import inference as ga_inference  # Importa la funzione di GA-DPAMSA
import pandas as pd
import config
import datasets.inference_dataset.encode_project_dataset_4x101bp as dataset1

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
sp_mafft_values = []
sp_muscle5_values = []
sp_pasta_values = []
sp_upp_values = []

# Crea la directory per gli output di altri tool se necessario
output_folders = {
    'clustalw': './clustalw_output',
    'mafft': './mafft_output',
    'muscle5': './muscle5_output',
    'upp': './upp_output',
    'pasta': './pasta_output'
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
    command_mafft = ['mafft', '--auto', file_path]
    command_muscle5 = ['muscle5', '-align', file_path, '-output', f'./muscle5_output/{file}']
    command_upp = ['run_upp.py', '-s', file_path, '-m', 'dna', '-d', f'./upp_output/{file}_output']
    command_pasta = ['run_pasta.py', '-i', file_path, '-o', f'./pasta_output/{file}_output']

    # Esegui Clustal Omega
    alignment_output_clustalo = subprocess.run(command_clustalo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout

    # Esegui MSAProbs
    alignment_output_msaprobs = subprocess.run(command_msaprobs, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout

    # Esegui ClustalW
    subprocess.run(command_clustalw, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    with open(f'./clustalw_output/{file}', 'r') as f:
        alignment_output_clustalw = f.read()

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

    # Esegui PASTA
    subprocess.run(command_pasta, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    file_name = os.path.splitext(file)[0]
    with open(f'./pasta_output/{file}_output/pastajob.marker001.{file_name}.aln', 'r') as f:
        alignment_output_pasta = f.read()

    # Calcola i Sum of Pairs per ogni tool
    sum_of_pairs_clustalo = utils.sum_of_pairs_from_fasta(alignment_output_clustalo)
    sum_of_pairs_msaprobs = utils.sum_of_pairs_from_fasta(alignment_output_msaprobs)
    sum_of_pairs_clustalw = utils.sum_of_pairs_from_fasta(alignment_output_clustalw)
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
        sum_of_pairs_mafft,
        sum_of_pairs_muscle5,
        sum_of_pairs_pasta,
        sum_of_pairs_upp
    ])

    # Aggiungi i valori alla lista per il calcolo delle medie
    sp_clustalo_values.append(sum_of_pairs_clustalo)
    sp_msaprobs_values.append(sum_of_pairs_msaprobs)
    sp_clustalw_values.append(sum_of_pairs_clustalw)
    sp_mafft_values.append(sum_of_pairs_mafft)
    sp_muscle5_values.append(sum_of_pairs_muscle5)
    sp_pasta_values.append(sum_of_pairs_pasta)
    sp_upp_values.append(sum_of_pairs_upp)

# **Esegui GA-DPAMSA una sola volta dopo il ciclo**
ga_inference(dataset=dataset1, model_path='model_3x30', truncate_file=True)

# **Leggi il CSV generato da GA-DPAMSA**
ga_csv_path = os.path.join(utils.CSV_PATH, f"{dataset_name}.csv")
ga_data = pd.read_csv(ga_csv_path)

# **Aggiungi la colonna SP Score di GA-DPAMSA ai dati esistenti**
ga_sp_scores = ga_data['SP Score'].tolist()

# **Assicurati che la lunghezza dei dati corrisponda**
if len(ga_sp_scores) == len(csv_data):
    for i in range(len(csv_data)):
        csv_data[i].append(ga_sp_scores[i])
else:
    print("Errore: Il numero di risultati GA-DPAMSA non corrisponde al numero di file processati.")

# Salva i risultati delle SP per ogni file
with open(f'{utils.CSV_PATH}/{dataset_name}_tools_sp_score.csv', mode='w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    writer.writerow([
        "File name",
        "ClustalOmega",
        "MSAProbs",
        "ClustalW",
        "MAFFT",
        "MUSCLE5",
        "PASTA",
        "UPP",
        "GA-DPAMSA"
    ])
    writer.writerows(csv_data)

# Calcola le medie delle SP per ogni tool
average_sp_clustalo = sum(sp_clustalo_values) / len(sp_clustalo_values)
average_sp_msaprobs = sum(sp_msaprobs_values) / len(sp_msaprobs_values)
average_sp_clustalw = sum(sp_clustalw_values) / len(sp_clustalw_values)
average_sp_mafft = sum(sp_mafft_values) / len(sp_mafft_values)
average_sp_muscle5 = sum(sp_muscle5_values) / len(sp_muscle5_values)
average_sp_pasta = sum(sp_pasta_values) / len(sp_pasta_values)
average_sp_upp = sum(sp_upp_values) / len(sp_upp_values)
average_sp_ga_dpamsa = sum(ga_sp_scores) / len(ga_sp_scores)

# Salva i risultati medi in un altro file CSV
with open(f'{utils.CSV_PATH}/{dataset_name}_tools_average_sp.csv', mode='w', newline='') as avg_csv:
    writer = csv.writer(avg_csv)
    writer.writerow(["Tool", "Average SP"])
    writer.writerow(["ClustalOmega", average_sp_clustalo])
    writer.writerow(["MSAProbs", average_sp_msaprobs])
    writer.writerow(["ClustalW", average_sp_clustalw])
    writer.writerow(["MAFFT", average_sp_mafft])
    writer.writerow(["MUSCLE5", average_sp_muscle5])
    writer.writerow(["PASTA", average_sp_pasta])
    writer.writerow(["UPP", average_sp_upp])
    writer.writerow(["GA-DPAMSA", average_sp_ga_dpamsa])


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

### DATA VISUALIZATION ###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Carica i dati dal file CSV
data = pd.read_csv(f'{utils.CSV_PATH}/{dataset_name}_tools_sp_score.csv')

# Riorganizza i dati in formato "long" per facilitare la visualizzazione con seaborn
data_long = pd.melt(data, id_vars=["File name"],
                    value_vars=["ClustalOmega",
                                "MSAProbs",
                                "ClustalW",
                                "MAFFT",
                                "MUSCLE5",
                                "PASTA",
                                "UPP",
                                "GA-DPAMSA"],
                    var_name='Tool', value_name='SP_Score')

# Configura il tema di seaborn per uno stile più pulito
sns.set(style="whitegrid")

# Colori personalizzati per i tool
palette = sns.color_palette("Set2", len(data_long['Tool'].unique()))

### BoxPlot per i dati generali di SP score ###
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='Tool', y='SP_Score', data=data_long, palette=palette)

# Imposta il colore dei contorni più scuro e il riempimento più trasparente
for patch, color in zip(box_plot.patches, palette):
    # Riempimento con trasparenza
    transparent_fill = mcolors.to_rgba(color, alpha=0.5)  # Imposta un'alpha per il riempimento
    patch.set_facecolor(transparent_fill)

    # Imposta il colore per i bordi
    patch.set_edgecolor(color)
    patch.set_linewidth(1.5)

# Imposta il colore delle linee (mediana, whiskers, ecc.) usando i colori della palette
for i, line in enumerate(box_plot.lines):
    # Ogni box ha 6 linee (mediana, whiskers, caplines)
    color = palette[i // 6 % len(palette)]  # Associa il colore corretto a ogni set di linee
    line.set_color(color)
    line.set_linewidth(1.5)

plt.title('Distribuzione degli SP Scores per ogni Tool')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{utils.CHARTS_PATH}/{dataset_name}_boxplot.png')  # Salva il grafico come immagine
plt.show()

### BarPlot per i valori medi di SP score ###
# Calcola i valori medi dal DataFrame
mean_sp_scores = data_long.groupby('Tool')['SP_Score'].mean().reset_index()

plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Tool', y='SP_Score', data=mean_sp_scores, palette=palette)

# Aggiungi i valori medi sopra le barre
for index, row in mean_sp_scores.iterrows():
    bar_plot.text(index, row.SP_Score + 0.01, round(row.SP_Score, 2), color='black', ha="center")

plt.title('Valori Medi degli SP Scores per ogni Tool')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{utils.CHARTS_PATH}/{dataset_name}_barplot.png')  # Salva il grafico come immagine
plt.show()
