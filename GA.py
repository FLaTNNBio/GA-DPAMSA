from DPAMSA.env import Environment
from DPAMSA.dqn import DQN
import config
import utils
import copy
import random


nucleotides_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4, '-': 5}
nucleotides = ['A', 'T', 'C', 'G', '-']

class GA:
    def __init__(self,sequences):
        self.sequences = sequences
        self.population_size = config.GA_POPULATION_SIZE
        self.population = []
        self.population_column_score = []
        self.population_score = []
        self.unique_ranges = []

    #Generate population for the Genetic Algorithm
    def generate_population(self):
        for i in range(self.population_size):
            self.population.append([[nucleotides_map[self.sequences[i][j]] for j in range(len(self.sequences[i]))] for i in range(len(self.sequences))])
        self.unique_ranges = utils.get_all_different_sub_range(self.population[0],config.AGENT_WINDOW_ROW,config.AGENT_WINDOW_COLUMN)

    #Sum-of-pairs
    def calculate_fitness_score(self):
        self.population_score = []
        for index_chromosome,chromosome in enumerate(self.population):
            #When RL is applied only on a sub-board, some sequences may become longer because of gaps
            #then gaps are added at the end of all sequences before the sum-of-pairs calculation
            #can happen that RL agent goes in a sub-board where there are some holes?
            gene_max_len = max(len(gene) for gene in chromosome)
            for gene in chromosome: 
                while len(gene) < gene_max_len:
                    gene.append(5)
        
            num_sequences = len(chromosome)
            score = 0
            for i in range(len(chromosome[0])):
                for j in range(num_sequences):
                    for k in range(j + 1, num_sequences):
                        if chromosome[j][i] == 5 or chromosome[k][i] == 5:
                            score += config.GAP_PENALTY
                        elif chromosome[j][i] == chromosome[k][i]:
                            score += config.MATCH_REWARD
                        elif chromosome[j][i] != chromosome[k][i]:
                            score += config.MISMATCH_PENALTY
            self.population_score.append((index_chromosome,score))

    def calculate_column_score(self):
        self.population_column_score = []

        for index_chromosome, chromosome in enumerate(self.population):
            gene_max_len = max(len(gene) for gene in chromosome)
            for gene in chromosome:
                while len(gene) < gene_max_len:
                    gene.append(5)  # gap

            num_sequences = len(chromosome)
            num_columns = gene_max_len
            column_score_count = 0

            for col in range(num_columns):
                col_values = [chromosome[row][col] for row in range(num_sequences)]
                if all(x == col_values[0] for x in col_values):
                    column_score_count += 1

            column_score = column_score_count / num_columns

            self.population_column_score.append((index_chromosome, column_score))
    
    def selection(self, column_score_mode=False):
        #Selection
        #Sort the population based on the score
        if column_score_mode:
            population_score_sorted = sorted(self.population_column_score, key=lambda x: x[1])
        else:
            population_score_sorted = sorted(self.population_score, key=lambda x: x[1])
        #Get the index of the worst fitted individuals
        worst_fitted_individual = [item[0] for item in population_score_sorted[:config.GA_NUM_MOST_FIT_FOR_ITER]]
        #Delete individuals with the worst score 
        for index in sorted(worst_fitted_individual,reverse=True):
            self.population.pop(index)


    def selection_intersection(self):
        # Calcola entrambe le metriche per la popolazione corrente
        self.calculate_fitness_score()
        self.calculate_column_score()

        # Ordina gli individui in base al punteggio sum-of-pairs (migliori = punteggio più alto)
        sorted_sop = sorted(self.population_score, key=lambda x: x[1], reverse=True)
        # Ordina gli individui in base al column score (migliori = punteggio più alto)
        sorted_column = sorted(self.population_column_score, key=lambda x: x[1], reverse=True)

        # Definisci quanti individui considerare tra i migliori per ciascuna metrica
        top_n = config.GA_NUM_MOST_FIT_FOR_ITER

        # Estrai gli indici dei top individui per ciascuna metrica
        top_indices_sop = {ind for ind, score in sorted_sop[:top_n]}
        top_indices_column = {ind for ind, score in sorted_column[:top_n]}

        # Calcola l'intersezione degli indici
        intersection_indices = top_indices_sop.intersection(top_indices_column)

        # Se l'intersezione è vuota, come fallback prendi la loro unione
        if not intersection_indices:
            intersection_indices = top_indices_sop.union(top_indices_column)

        # Aggiorna la popolazione mantenendo solo gli individui dell'intersezione
        new_population = [chromosome for idx, chromosome in enumerate(self.population) if idx in intersection_indices]
        self.population = new_population
        
    
    def get_alignment(self,chromosome):
        alignment = ""
        for i in range(len(chromosome)):
            alignment += ''.join([nucleotides[chromosome[i][j] - 1] for j in range(len(chromosome[i]))]) + '\n'

        return alignment.rstrip()
    
    def get_most_fitted_chromosome(self, column_score_mode):
        #Sort the population based on the score
        if not column_score_mode:
            population_score_sorted = sorted(self.population_score, key=lambda x: x[1], reverse=True)
        else:
            population_score_sorted = sorted(self.population_column_score, key=lambda x: x[1], reverse=True)
        most_fitted_individual = self.population[population_score_sorted[0][0]]     
        #Clean all gaps that appear after the last nucleotide (if along the whole row and all columns there are only gaps)
        utils.clean_unnecessary_gaps(most_fitted_individual)
        # final_score = utils.get_sum_of_pairs(most_fitted_individual,0,len(most_fitted_individual),0,len(most_fitted_individual[0]))
        # return most_fitted_individual,final_score
        return most_fitted_individual


    def get_most_fitted_chromosome_intersection(self):
        # Calcola entrambe le metriche per la popolazione
        self.calculate_fitness_score()
        self.calculate_column_score()

        # Calcola un punteggio combinato per ogni individuo (somma del sum-of-pairs e del column score)
        combined_scores = []
        for idx in range(len(self.population)):
            sop_score = next(score for i, score in self.population_score if i == idx)
            col_score = next(score for i, score in self.population_column_score if i == idx)
            combined_scores.append((idx, sop_score + col_score))

        # Seleziona l'individuo con il punteggio combinato più alto
        best_idx, best_combined = max(combined_scores, key=lambda x: x[1])
        best_individual = self.population[best_idx]

        # Pulisci eventuali gap non necessari (come fatto in get_most_fitted_chromosome)
        utils.clean_unnecessary_gaps(best_individual)

        # Calcola i punteggi finali per il cromosoma migliore
        final_sum_pairs = utils.get_sum_of_pairs(
            best_individual,
            0,
            len(best_individual),
            0,
            len(best_individual[0])
        )
        final_column_score = utils.get_column_score(
            best_individual,
            0,
            len(best_individual),
            0,
            len(best_individual[0])
        )

        return best_individual, final_sum_pairs, final_column_score

    def vertical_crossover(self):
        #Calculation of the mean length of a sequences, to calculate the position in which we cut every sequence in a chromosome
        number_of_nucleotides = []
        for genes in self.population[0]:
            number_of_nucleotides.append(len(genes))
        mean_length = int((sum(number_of_nucleotides) / len(number_of_nucleotides)) / 2)
        
        #Crossover
        new_individuals = []
        while (len(self.population) + len(new_individuals) < config.GA_POPULATION_SIZE): #Repeat two times to have a costant number of population (with one iteration we generate only the half of GA_NUM_MOST_FIT_FOR_ITER individuals)
            

            #for i in range(0, len(self.population) - 1,2):
            index_parent1 = random.randint(0,len(self.population) - 1)
            index_parent2 = random.randint(0,len(self.population) - 1)
            parent1 = self.population[index_parent1]
            parent2 = self.population[index_parent2]
            first_half_parent1 = []
            second_half_parent2 = []

            #Calculation of the mean length of a sequences, to calculate the position in which we cut every sequence in a chromosome
            number_of_nucleotides = []
            for genes in parent1:
                number_of_nucleotides.append(len(genes))
            mean_length_parent1 = int((sum(number_of_nucleotides) / len(number_of_nucleotides)) / 2)

            number_of_nucleotides = []
            for genes in parent2:
                number_of_nucleotides.append(len(genes))
            mean_length_parent2 = int((sum(number_of_nucleotides) / len(number_of_nucleotides)) / 2)


            #First half of genes from parent1
            for genes in parent1:
                first_half = genes[:mean_length_parent1]
                first_half_parent1.append(first_half)

            #Second half of genes from parent2
            for genes in parent2:
                second_half = genes[mean_length_parent2:]
                second_half_parent2.append(second_half)
            
            #Contruct the new individual
            new_chromosome = []
            for k in range(len(first_half_parent1)):
                new_chromosome.append(first_half_parent1[k] + second_half_parent2[k])
            new_individuals.append(new_chromosome)

        #Update the population with new individals
        new_population = self.population + new_individuals
        self.population = new_population
        
        return 
    
    def horizontal_crossover(self):
        num_seq = len(self.population[0])

        #Check if the number of sequence is even (I do not break exactly into two equal parts)
        if num_seq % 2 == 0:
            cut_index = num_seq // 2
        else:
            cut_index = (num_seq // 2) + 1
        
        new_indivisuals = []
        while (len(self.population) + len(new_indivisuals) < config.GA_POPULATION_SIZE): #Repeat until we reach again the number of desidered individual in the population
            #for i in range(0, len(self.population) - 1,2): #Loop on population in steps of 2
                index_parent1 = random.randint(0,len(self.population) - 1)
                index_parent2 = random.randint(0,len(self.population) - 1)
                parent1 = self.population[index_parent1]
                parent2 = self.population[index_parent2]
                first_half_parent1 = []
                second_half_parent2 = []

                #First half of genes from parent1
                first_half_parent1 = parent1[:cut_index]
                #Second half of genes from parent1
                second_half_parent2 = parent2[cut_index:]

                #Contruct the new individual
                new_chromosome = first_half_parent1 + second_half_parent2
                new_indivisuals.append(new_chromosome)
        
        new_population = self.population + new_indivisuals
        self.population = new_population
        
        return

    #Perform gene mutation for random selected individuals
    def random_mutation(self,model_path):
        #The mutation is performed until we cover all the possible sub-board for a individual
        selected_individual_index = utils.casual_number_generation(0, self.population_size - 1, len(self.unique_ranges))
        ranges_for_iterations = copy.deepcopy(self.unique_ranges)
        for index in selected_individual_index:
            individual_to_mutate = self.population[index]

            #Construct the sub-board
            selected_range = random.choice(ranges_for_iterations)
            ranges_for_iterations.remove(selected_range)
            from_row, to_row, from_column, to_column = selected_range
            
            #Get only the selected row
            row_genes = individual_to_mutate[from_row:to_row]
            sub_board = []

            ##To prevent to fill the space with all gaps is better to have that the sub-board is a multiple of the main board in terms of row x column
            ##If the main board can't be perfectly divide in slice of size AGENT_WINDOW_ROW, a raw with all GAP is added to fill the space (the RL agent won't work if size is less than the size in the training)
            fake_row_counter = 0
            while (len(row_genes) < config.AGENT_WINDOW_ROW):
                all_gap_row = []
                while (len(all_gap_row) < config.AGENT_WINDOW_COLUMN):
                    all_gap_row.append(5)
                fake_row_counter = fake_row_counter + 1
                row_genes.append(all_gap_row)

            for genes in row_genes:
                sub_genes = genes[from_column:to_column]  
                #If the main board can't be perfectly divide in slice of size AGENT_WINDOW_COLUMN, GAP is added to fill the space (the RL agent won't work if size is less than the size in the training) 
                while len(sub_genes) < config.AGENT_WINDOW_COLUMN:
                    sub_genes.append(5)
                sub_board.append(sub_genes)

            #Perform Mutation on the sub-board with RL
            env = Environment(sub_board, convert_data=False)
            agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
            agent.load(model_path)
            state = env.reset()

            while True:
                action = agent.predict(state)
                _, next_state, done = env.step(action)
                state = next_state
                if 0 == done:
                    break
            
            env.padding()
            #Put mutated genes in the right position in the individual
            genes_to_mutate = individual_to_mutate[from_row:to_row]
            for index,sequence in enumerate(env.aligned):
                    #if(index < len(genes_to_mutate) - 1): #This is necessary due to the row with all GAP added in case the number of row for the window is not multiple of the main board rows
                    genes_to_mutate[index][from_column:to_column] = sequence
            individual_to_mutate[from_row:to_row] = genes_to_mutate

    #Perform gene mutation only on individuals with the highest sum-of-pairs-score and then apply the mutation on the worst sub-board
    def mutation_on_best_fitted_individuals_worst_sub_board(self,model_path, column_score_mode=False):
        #The mutation is performed until we cover all the possible sub-board for a individual
        if column_score_mode:
            self.calculate_column_score()
        else:
            self.calculate_fitness_score()
        num_individuals_to_mutate = round(config.GA_POPULATION_SIZE * config.GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER)
        if column_score_mode:
            best_fitted_individual = utils.get_index_of_the_best_fitted_individuals(self.population_column_score,
                                                                                    num_individuals_to_mutate)
        else:
            best_fitted_individual = utils.get_index_of_the_best_fitted_individuals(self.population_score,
                                                                                    num_individuals_to_mutate)
        for index in best_fitted_individual:
            individual_to_mutate = self.population[index]

            #Check the worst fitted sub-board based on the sum-of-pairs
            score, worst_fitted_range = utils.calculate_worst_fitted_sub_board(individual_to_mutate)
            from_row,to_row,from_column,to_column = worst_fitted_range

            #Get only the selected row
            row_genes = individual_to_mutate[from_row:to_row]
            sub_board = []

            ##To prevent to fill the space with all gaps is better to have that the sub-board is a multiple of the main board in terms of row x column
            ##If the main board can't be perfectly divide in slice of size AGENT_WINDOW_ROW, a raw with all GAP is added to fill the space (the RL agent won't work if size is less than the size in the training)
            fake_row_counter = 0
            while (len(row_genes) < config.AGENT_WINDOW_ROW):
                all_gap_row = []
                while (len(all_gap_row) < config.AGENT_WINDOW_COLUMN):
                    all_gap_row.append(5)
                fake_row_counter = fake_row_counter + 1
                row_genes.append(all_gap_row)

            for genes in row_genes:
                sub_genes = genes[from_column:to_column]  
                #If the main board can't be perfectly divide in slice of size AGENT_WINDOW_COLUMN, GAP is added to fill the space (the RL agent won't work if size is less than the size in the training) 
                while len(sub_genes) < config.AGENT_WINDOW_COLUMN:
                    sub_genes.append(5)
                sub_board.append(sub_genes)

            #Perform Mutation on the sub-board with RL
            env = Environment(sub_board, convert_data=False)
            agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
            agent.load(model_path)
            state = env.reset()

            while True:
                action = agent.predict(state)
                _, next_state, done = env.step(action)
                state = next_state
                if 0 == done:
                    break
            
            env.padding()
            #Put mutated genes in the right position in the individual
            genes_to_mutate = individual_to_mutate[from_row:to_row]
            for index,sequence in enumerate(env.aligned):
                    #if(index < len(genes_to_mutate) - 1): #This is necessary due to the row with all GAP added in case the number of row for the window is not multiple of the main board rows
                    genes_to_mutate[index][from_column:to_column] = sequence
            individual_to_mutate[from_row:to_row] = genes_to_mutate

    def get_nucleotides_seqs(self, chromosome):

        nucleotides_seqs = []
        for sequence in chromosome:
            # Converti ogni numero nel corrispondente nucleotide
            nucleotide_sequence = ''.join([nucleotides[n - 1] for n in sequence])
            nucleotides_seqs.append(nucleotide_sequence)

        return nucleotides_seqs