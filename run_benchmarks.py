import os
from tqdm import tqdm
import config
import utils
import datasets.inference_dataset.encode_project_dataset_4x101bp_2 as inference_dataset

# MAKE SURE THE DATASET NAME MATCHES THE IMPORTED DATASET
DATASET_NAME = 'encode_project_dataset_4x101bp'

# MAKE SURE TO CHANGE 'DATASET_ROW' AND 'DATASET_COLUMN' IN THE CONFIG FILE TO MATCH DATASET SIZES
# MAKE SURE DPAMSA MODEL MATCHES DATASET SIZES
DPAMSA_MODEL = 'model_3x30'

# MAKE SURE 'AGENT_WINDOW_ROW' AND 'AGENT_WINDOW_COLUMN' MATCH GA_DPAMSA MODEL
GA_DPAMSA_MODEL = 'model_3x30'


def main():
    # Mostra il menu di selezione
    choice = utils.display_menu()

    dataset_folder = os.path.join(config.FASTA_FILES_PATH, DATASET_NAME)
    file_paths = [os.path.join(dataset_folder, file) for file in sorted(os.listdir(dataset_folder))]

    # Dizionario per memorizzare i CSV dei tool usati
    tool_csv_paths = {}

    # GA-DPAMSA deve essere sempre eseguito
    ga_csv_path = utils.run_ga_dpamsa_inference(inference_dataset, DATASET_NAME, GA_DPAMSA_MODEL)
    tool_csv_paths['GA-DPAMSA'] = ga_csv_path

    # Esegui DPAMSA se l'utente sceglie l'opzione 1 o 3
    if choice == 1 or choice == 3:
        dpamsa_csv_path = utils.run_dpamsa_inference(inference_dataset, DATASET_NAME, DPAMSA_MODEL)
        tool_csv_paths['DPAMSA'] = dpamsa_csv_path

    # Esegui i Tool esterni se l'utente sceglie l'opzione 2 o 3
    if choice == 2 or choice == 3:
        for tool_name in tqdm(config.TOOLS.keys(), desc="Processing Tools"):
            tool_results = utils.run_tool_and_generate_report(tool_name, file_paths, DATASET_NAME)
            csv_path = utils.save_inference_csv(tool_results, tool_name, DATASET_NAME)  # Salva il CSV e ottieni il path
            tool_csv_paths[tool_name] = csv_path

    # Visualizza i risultati finali solo per i tool utilizzati
    utils.plot_metrics(tool_csv_paths, DATASET_NAME)


if __name__ == "__main__":
    main()
