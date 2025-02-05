import datasets.training_dataset.encode_project_dataset_4x101bp as dataset1
from GA import GA
import torch
import os
from env_GA import Environment
from dqn import DQN
import config
from tqdm import tqdm
import random
import utils

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

def inference(tag='',start=0, end=1, truncate_file=False, model_path='model', dataset=dataset):
    output_parameters()
    print("Dataset number: {}".format(len(dataset.datasets)))

    report_file_name = os.path.join(config.report_path_DPAMSA_GA, "{}.rpt".format(tag))

    if truncate_file:
        with open(report_file_name, 'w') as _:
            _.truncate()

    for dataset_name in dataset.datasets:
        
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
        #In the last iteration, we have to perform again the calculation (last operation is the crossover so we need to recheck the score) 
        ga.calculate_fitness_score()
        most_fitted_chromosome,sum_pairs_score = ga.get_most_fitted_chromosome()
        most_fitted_chromosome_converted = ga.get_alignment(most_fitted_chromosome)
        print(f"Dataset name: {dataset_name}")
        print(f"SP:{sum_pairs_score}")
        print(f"Alignment:\n{most_fitted_chromosome_converted}")
        report = f"Dataset name: {dataset_name}\nSP: {sum_pairs_score}\nAlignment:\n{most_fitted_chromosome_converted}\n\n"
        with open(os.path.join(config.report_path_DPAMSA_GA, "{}.rpt".format(tag)), 'a') as report_file:
            report_file.write(report)
            
if __name__ == "__main__":
    inference(model_path='model_3x30',tag=dataset.file_name)