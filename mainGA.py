import datasets.inference_dataset.dataset1_3x30bp as dataset1
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
    print("Episode: {}".format(config.max_episode))
    print("Batch size: {}".format(config.batch_size))
    print("Replay memory size: {}".format(config.replay_memory_size))
    print("Alpha: {}".format(config.alpha))
    print("Epsilon: {}".format(config.epsilon))
    print("Gamma: {}".format(config.gamma))
    print("Delta: {}".format(config.delta))
    print("Decrement iteration: {}".format(config.decrement_iteration))
    print("Update iteration: {}".format(config.update_iteration))
    print("Device: {}".format(config.device_name))
    print("-------- Genetic Algorithm parameters ---------")
    print(f"Window size:{config.AGENT_WINDOW_ROW}x{config.AGENT_WINDOW_COLUMN}")
    print(f"Population number: {config.GA_POPULATION_SIZE}")
    print(f"Number of iteration: {config.GA_NUM_ITERATION}")
    print(f"Number of most fitted individuals selected for iteration: {config.GA_NUM_MOST_FIT_FOR_ITER}")
    print(f"Percentage of individuals to mutate for iteration (only if is used mutation on the worst-fitted individuals): {config.GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER}")
    print('\n')

def inference(dataset=dataset, start=0, end=-1, model_path='model_3x30', truncate_file=True):

    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(config.report_path_GA_DPAMSA, f"{tag}.rpt")
    csv_file_name = os.path.join(config.csv_path, f"{tag}.csv")

    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # writer.writerow(["Dataset Name", "Number of Sequences", "Alignment Length", "SP Score", "Exact Matches",
            #                 "Column Score"])
            writer.writerow(["Dataset Name", "SP Score"])

    datasets_to_process = dataset.datasets[start:end if end != -1 else len(dataset.datasets)]

    for index, dataset_name in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):
        
        if not hasattr(dataset, dataset_name):
            continue
        seqs = getattr(dataset, dataset_name)

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
            ga.mutation_on_best_fitted_individuals_worst_sub_board(model_path)   
            #Calculate the fitness score for all individuals, based on the sum-of-pairs
            ga.calculate_fitness_score()

            #Execute the selection, get only the most fitted individual for the next iteration
            ga.selection()

            #Crossover, split board in two different part and create new individuals by merging each part by 
            #taking the first part from one individual and the second part from another individual
            ga.horizontal_crossover()
            #ga.vertical_crossover()
        #In the last iteration, we have to perform again the calculation (last operation is the crossover, so we need to recheck the score)
        ga.calculate_fitness_score()
        most_fitted_chromosome,sum_pairs_score = ga.get_most_fitted_chromosome()
        most_fitted_chromosome_converted = ga.get_alignment(most_fitted_chromosome)

        report = (
            f"#: {dataset_name}\n"
            f"SP: {sum_pairs_score}\n"
            f"Alignment:\n{most_fitted_chromosome_converted}\n\n"
        )

        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([dataset_name, sum_pairs_score])

    print(f"\nOperazione completata con successo.")
    print(f"Il file di report è stato salvato in: {config.report_path_GA_DPAMSA}")
    print(f"Il file CSV è stato salvato in: {config.csv_path}")


if __name__ == "__main__":
    inference(model_path='model_3x30')