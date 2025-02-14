import datasets.inference_dataset.dataset1_3x30bp as dataset1
import utils
from DPAMSA.env import Environment
from GA import GA
import config
import csv
import os
from tqdm import tqdm


dataset = dataset1


def output_parameters():
    print("---------- DPAMSA parameters -----------------")
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.MAX_EPISODE))
    print("Batch size: {}".format(config.BATCH_SIZE))
    print("Replay memory size: {}".format(config.REPLAY_MEMORY_SIZE))
    print("Alpha: {}".format(config.ALPHA))
    print("Epsilon: {}".format(config.EPSILON))
    print("Gamma: {}".format(config.GAMMA))
    print("Delta: {}".format(config.DELTA))
    print("Decrement iteration: {}".format(config.DECREMENT_ITERATION))
    print("Update iteration: {}".format(config.UPDATE_ITERATION))
    print("Device: {}".format(config.DEVICE_NAME))
    print("-------- Genetic Algorithm parameters ---------")
    print(f"Window size:{config.AGENT_WINDOW_ROW}x{config.AGENT_WINDOW_COLUMN}")
    print(f"Population number: {config.GA_POPULATION_SIZE}")
    print(f"Number of iteration: {config.GA_NUM_ITERATION}")
    print(f"Number of most fitted individuals selected for iteration: {config.GA_NUM_MOST_FIT_FOR_ITER}")
    print(f"Percentage of individuals to mutate for iteration (only if is used mutation on the worst-fitted individuals): {config.GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER}")
    print('\n')


def inference(dataset=dataset, start=0, end=-1, model_path='model_3x30', truncate_file=True,  column_score_mode=False, multi_objective_mode=True):
    if column_score_mode and multi_objective_mode:
        raise Exception("You can't use both column score mode and multi objective mode")
    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(config.GA_DPAMSA_REPORTS_PATH, f"{tag}.txt")
    csv_file_name = os.path.join(config.GA_DPAMSA_INF_CSV_PATH, f"{tag}_GA_DPAMSA_results.csv")

    if truncate_file:
        # Se truncate_file è True, tronca i file
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                             "Exact Matches (EM)", "Column Score (CS)"])
    else:
        # Se truncate_file è False, crea il file se non esiste e scrivi l'intestazione
        if not os.path.exists(report_file_name):
            with open(report_file_name, 'w'):
                pass
        if not os.path.exists(csv_file_name):
            with open(csv_file_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["File Name", "Number of Sequences (QTY)", "Alignment Length (AL)", "Sum of Pairs (SP)",
                                 "Exact Matches (EM)", "Column Score (CS)"])

    datasets_to_process = dataset.datasets[start:end if end != -1 else len(dataset.datasets)]

    for index, dataset_name in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):
        
        if not hasattr(dataset, dataset_name):
            continue
        seqs = getattr(dataset, dataset_name)

        env = Environment(seqs, convert_data=False)
        ga = GA(seqs)
        ga.generate_population()

        #Check the correct size of the window in which the RL agent operate
        first_individual = ga.population[0]
        if config.AGENT_WINDOW_ROW > len(first_individual):
            print("Window row grater than the number of a sequence")

            return 
        
        sequence_min_length = min(len(genes) for genes in first_individual)
        if config.AGENT_WINDOW_COLUMN > sequence_min_length:
            print("Window column grater than the min length of a sequence")

            return 
        
        for i in range(config.GA_NUM_ITERATION):
        
            #Mutation with the RL agent
            #ga.random_mutation(model_path)
            ga.mutation_on_best_fitted_individuals_worst_sub_board(model_path, column_score_mode)
            # Calculate the fitness score for all individuals, based on the sum-of-pairs
            if not column_score_mode:
                ga.calculate_fitness_score()
            else:
                ga.calculate_column_score()
            # Execute the selection, get only the most fitted individual for the next iteration
            if not multi_objective_mode:
                ga.selection(column_score_mode)
            else:
                ga.selection_intersection()

            #Crossover, split board in two different part and create new individuals by merging each part by 
            #taking the first part from one individual and the second part from another individual
            ga.horizontal_crossover()
            #ga.vertical_crossover()
        #In the last iteration, we have to perform again the calculation (last operation is the crossover, so we need to recheck the score)
        if not column_score_mode:
            ga.calculate_fitness_score()
        else:
            ga.calculate_column_score()

        if not multi_objective_mode:
            most_fitted_chromosome = ga.get_most_fitted_chromosome(column_score_mode)
        else:
            most_fitted_chromosome, sum_pairs_score, final_column_score = ga.get_most_fitted_chromosome_intersection()
        #print(most_fitted_chromosome)
        #most_fitted_chromosome = ga.get_most_fitted_chromosome()
        aligned_seqs = ga.get_nucleotides_seqs(most_fitted_chromosome)

        Environment.set_alignment(env, aligned_seqs)
        metrics = utils.calculate_metrics(env)

        # Crea il report testuale
        report = (
            f"File: {dataset_name}\n"
            f"Alignment Length (AL): {metrics['AL']}\n"
            f"Number of Sequences (QTY): {metrics['QTY']}\n"
            f"Sum of Pairs (SP): {metrics['SP']}\n"
            f"Exact Matches (EM): {metrics['EM']}\n"
            f"Column Score (CS): {metrics['CS']:.3f}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        # Scrive il report nel file .rpt
        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        # Scrive i dati nel file CSV
        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([dataset_name, metrics['AL'], metrics['QTY'], metrics['SP'], metrics['EM'], metrics['CS']])


    print(f"\nOperazione completata con successo.")
    print(f"Il file di report è stato salvato in: {config.GA_DPAMSA_REPORTS_PATH}")
    print(f"Il file CSV è stato salvato in: {config.CSV_PATH}\n\n")


if __name__ == "__main__":
    inference(model_path='model_3x30')