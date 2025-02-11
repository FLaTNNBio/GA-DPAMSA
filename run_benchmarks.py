import os
from tqdm import tqdm
import utils
import datasets.inference_dataset.encode_project_dataset_4x101bp as inference_dataset

# MAKE SURE THE DATASET NAME MATCHES THE IMPORTED DATASET
DATASET_NAME = 'encode_project_dataset_4x101bp'  # Fasta Files Dataset
DPAMSA_MODEL = 'model_3x30'
GA_DPAMSA_MODEL = 'model_3x30'


def main():
    # Mostra il menu di selezione
    choice = utils.display_menu()

    dataset_folder = os.path.join(utils.FASTA_FILES_PATH, DATASET_NAME)
    file_paths = [os.path.join(dataset_folder, file) for file in sorted(os.listdir(dataset_folder))]

    main_csv_path = os.path.join(utils.CSV_PATH, 'tools_metrics.csv')
    additional_csv_paths = []

    # **GA-DPAMSA deve essere sempre eseguito**
    ga_csv_path = utils.run_ga_dpamsa_inference(inference_dataset, DATASET_NAME, GA_DPAMSA_MODEL)
    additional_csv_paths.append(ga_csv_path)

    # Esegui DPAMSA se l'utente sceglie l'opzione 1 o 3
    if choice == 1 or choice == 3:
        dpamsa_csv_path = utils.run_dpamsa_inference(inference_dataset, DATASET_NAME, DPAMSA_MODEL)
        additional_csv_paths.append(dpamsa_csv_path)

    # Esegui i Tool esterni se l'utente sceglie l'opzione 2 o 3
    if choice == 2 or choice == 3:
        for tool_name in tqdm(utils.TOOLS.keys(), desc="Processing Tools"):
            tool_results = utils.run_tool_and_generate_report(tool_name, file_paths, DATASET_NAME)
            utils.append_results_to_csv(tool_results)

    # Aggrega i dati nel CSV unico
    if additional_csv_paths:
        utils.aggregate_csvs(main_csv_path, additional_csv_paths)

    # Visualizza i risultati finali
    utils.plot_metrics(main_csv_path, DATASET_NAME)


if __name__ == "__main__":
    main()
