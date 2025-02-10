import config
import random
import io
import sys
import os


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


def sum_of_pairs_from_fasta(fasta_content):
    sequences = {}
    max_length = 0

    nucleotides_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4, '-': 5}

    # Simulate reading from a FASTA content string using io.StringIO
    with io.StringIO(fasta_content) as f:
        lines = f.readlines()
        seq = ''
        seq_name = ''
        for line in lines:
            if line.startswith('>'):
                if seq_name != '':
                    sequences[seq_name] = seq
                    max_length = max(max_length, len(seq))
                seq_name = line.strip().lstrip('>')
                seq = ''
            else:
                seq += line.strip()
        if seq_name != '':
            sequences[seq_name] = seq
            max_length = max(max_length, len(seq))

    # Align sequence in the list
    aligned_sequences = {}
    for seq_name, seq in sequences.items():
        aligned_seq = seq + '-' * (max_length - len(seq))
        aligned_sequences[seq_name] = aligned_seq

    # Create matrix
    matrix = []
    for seq_name in sorted(aligned_sequences.keys()):  # Sort sequence
        row = []
        for char in aligned_sequences[seq_name]:
            row.append(nucleotides_map.get(char, 0))  # Default 0 if char not in the map
        matrix.append(row)

    sum_of_pairs = get_sum_of_pairs(matrix,0,len(matrix),0,len(matrix[0]))

    return sum_of_pairs