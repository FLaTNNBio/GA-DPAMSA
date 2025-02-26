from DPAMSA.env import Environment
from DPAMSA.dqn import DQN
import config
import utils
import copy
import random


nucleotides_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4, '-': 5}


class GA:
    def __init__(self, sequences, mode):
        self.sequences = sequences
        self.mode = mode
        self.population_size = config.POPULATION_SIZE
        self.population = []
        self.population_score = []
        self.hall_of_fame = []
        self.current_iteration = 0

    def generate_population(self):
        """
        Generates the initial population with a mix of exact copies and slightly modified versions.

        - A small percentage of individuals are exact copies of the dataset.
        - The remaining individuals are modified by adding a small number of random gaps (~5% of the sequence length).

        This improves population diversity while ensuring that high-quality sequences are available.

        Returns:
            None
        """
        self.population = []  # Reset population

        # Determine the number of exact copies and modified individuals
        num_exact_copies = round(self.population_size * config.CLONE_RATE)
        num_modified = self.population_size - num_exact_copies

        # Add exact copies of dataset sequences
        for _ in range(num_exact_copies):
            self.population.append([[nucleotides_map[nuc] for nuc in seq] for seq in self.sequences])

        # Add modified sequences with random gaps
        for _ in range(num_modified):
            modified_individual = []
            for sequence in self.sequences:
                modified_seq = [nucleotides_map[nuc] for nuc in sequence]  # Convert to numerical form
                num_gaps = max(1, round(len(modified_seq) * config.GAP_RATE))  # 5% gaps

                # Insert gaps at random positions
                for _ in range(num_gaps):
                    gap_pos = random.randint(0, len(modified_seq) - 1)
                    modified_seq.insert(gap_pos, 5)  # Insert gap (5)

                modified_individual.append(modified_seq)

            self.population.append(modified_individual)

    def update_hall_of_fame(self):
        """
        Updates the Hall of Fame (HoF) with the best individual found so far.

        - If the HoF is empty, stores the best individual from the current population.
        - Otherwise, adds the HoF individual to a copy of the population and reevaluates the best.
        - Uses `get_index_of_the_best_fitted_individuals` to determine the absolute best.

        Returns:
            None
        """
        # If HoF is empty, initialize it with the best individual
        if not self.hall_of_fame:
            best_idx = utils.get_index_of_the_best_fitted_individuals(self.population_score, num_individuals=1)[0]
            best_individual = copy.deepcopy(self.population[best_idx])
            best_score = copy.deepcopy(self.population_score[best_idx])
            self.hall_of_fame = (best_individual, best_score)
            return

        # Create copies of the population and scores, including the Hall of Fame individual
        temp_population = copy.deepcopy(self.population)
        temp_scores = copy.deepcopy(self.population_score)

        hof_individual, hof_score = copy.deepcopy(self.hall_of_fame)
        temp_population.append(hof_individual)

        # The following code avoids HoF update issues
        # Appending just hof_score leads to wrong HoF update due to the saved index in hof_score
        if len(temp_scores[0]) == 2:
            # If only one metric is used (SP or CS)
            temp_scores.append((len(temp_scores), hof_score[1]))
        else:
            # Multi-Objective mode
            temp_scores.append((len(temp_scores), hof_score[1], hof_score[2]))

        # Find the absolute best individual among both the current population and the HoF individual
        best_idx = utils.get_index_of_the_best_fitted_individuals(temp_scores, num_individuals=1)[0]
        self.hall_of_fame = (copy.deepcopy(temp_population[best_idx]), copy.deepcopy(temp_scores[best_idx]))

    def calculate_fitness_score(self):
        self.population_score = []

        for index_chromosome, chromosome in enumerate(self.population):
            # Get max length of sequences in this chromosome
            max_length = max(map(len, chromosome))

            # Pad sequences to max length
            for gene in chromosome:
                gene.extend([5] * (max_length - len(gene)))  # Extend with gaps instead of a loop

            utils.clean_unnecessary_gaps(chromosome)
            max_length = max(map(len, chromosome))

            sp_score = utils.get_sum_of_pairs(chromosome, 0, len(chromosome), 0, max_length) \
                if self.mode in {'sp', 'mo'} else None
            cs_score = utils.get_column_score(chromosome, 0, len(chromosome), 0, max_length) \
                if self.mode in {'cs', 'mo'} else None

            if self.mode == 'mo':
                self.population_score.append((index_chromosome, sp_score, cs_score))
            else:
                self.population_score.append((index_chromosome, sp_score or cs_score))

        self.update_hall_of_fame()

    def _find_pareto_front(self):
        """
        Trova il Pareto Front basato sui punteggi SP e CS.
        Un individuo è nel Pareto Front se **non è dominato da nessun altro**.

        Un individuo `A` domina `B` se:
          - A ha un punteggio SP >= B **e**
          - A ha un punteggio CS >= B **e almeno uno è strettamente maggiore**

        :return: Lista di indici degli individui nel Pareto Front.
        """
        pareto_front = []

        for i, (idx1, sp1, cs1) in enumerate(self.population_score):
            dominated = False

            for j, (idx2, sp2, cs2) in enumerate(self.population_score):
                if i != j:
                    # Controlla se l'individuo `i` è dominato dall'individuo `j`
                    if (sp2 >= sp1 and cs2 >= cs1) and (sp2 > sp1 or cs2 > cs1):
                        dominated = True
                        break  # Se è dominato, non può stare nel Pareto Front

            if not dominated:
                pareto_front.append((idx1, sp1, cs1))  # Aggiunge al Pareto Front

        return pareto_front

    def selection(self):

        top_n = int(self.population_size * config.SELECTION_RATE)

        if self.mode in {'sp', 'cs'}:
            # Selezione standard basata su una singola metrica
            sorted_population = sorted(self.population_score, key=lambda x: x[1],
                                       reverse=True)  # Ordinamento decrescente
            top_indices = {ind for ind, score in sorted_population[:top_n]}

        elif self.mode == 'mo':
            # Trova il Pareto Front
            pareto_front = self._find_pareto_front()
            pareto_indices = {ind for ind, _, _ in pareto_front}

            # Se il Pareto Front è troppo piccolo, aggiungiamo altri individui
            if len(pareto_indices) < top_n:
                sorted_sop = sorted(self.population_score, key=lambda x: x[1], reverse=True)  # Ordina per SP
                sorted_column = sorted(self.population_score, key=lambda x: x[2], reverse=True)  # Ordina per CS

                # Prendi gli individui migliori da entrambe le metriche
                top_indices_sop = {ind for ind, _, _ in sorted_sop[:top_n]}
                top_indices_column = {ind for ind, _, _ in sorted_column[:top_n]}

                # Unione tra Pareto Front e migliori SP/CS (se necessario)
                pareto_indices = pareto_indices.union(top_indices_sop).union(top_indices_column)
            else:
                sorted_pareto = sorted(pareto_front, key=lambda x: (x[1], x[2]), reverse=True)
                pareto_indices = {ind for ind, _, _ in sorted_pareto[:top_n]}

            top_indices = pareto_indices  # Gli individui selezionati sono quelli in `top_indices`

        # Aggiorna la popolazione mantenendo solo gli individui selezionati
        self.population = [chromosome for idx, chromosome in enumerate(self.population) if idx in top_indices]
        self.calculate_fitness_score()

    def horizontal_crossover(self):
        """
        Performs horizontal crossover to generate new individuals.
        Each new individual is created by combining deep copies of sequences from two parents.
        The new individual takes the top part from one parent and the bottom part from the other.
        New children are generated until the population reaches config.POPULATION_SIZE.
        """
        new_individuals = []

        # Continue generating children until we reach the desired population size.
        while len(self.population) + len(new_individuals) < config.POPULATION_SIZE:
            # Randomly choose two distinct parents.
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            if parent1 is parent2:
                continue  # Ensure distinct parents

            num_seq = len(parent1)
            if num_seq < 2:
                continue  # Crossover isn't applicable if there's only one sequence

            # Choose a random crossover point between 1 and num_seq-1.
            cut_index = random.randint(1, num_seq - 1)

            # Create a new individual by combining deep copies of the parent's parts.
            child = copy.deepcopy(parent1[:cut_index]) + copy.deepcopy(parent2[cut_index:])
            new_individuals.append(child)

        # Add the new individuals to the population and update fitness scores.
        self.population.extend(new_individuals)
        self.calculate_fitness_score()

    def mutation(self, model_path):
        """
        Applies mutation to the best-fitted individuals by modifying their worst-performing sub-board.

        Mutation is performed using a reinforcement learning (RL) agent on the identified worst sub-region
        of the selected best individuals.

        Steps:
        1. Calculate the fitness scores for all individuals in the population.
        2. Select the top-performing individuals based on the configured mutation rate.
        3. Identify the worst-fitted sub-board (sub-region with the lowest fitness).
        4. Perform mutation using the RL agent on the worst sub-board.
        5. Replace the original sub-board with the mutated version.

        Args:
            model_path (str): Path to the trained RL model used for mutation.

        Returns:
            None
        """
        # Compute fitness scores based on the selected mode ('sp', 'cs', or 'mo')
        # self.calculate_fitness_score()

        # Determine the number of individuals to mutate
        num_individuals_to_mutate = round(
            config.POPULATION_SIZE * config.MUTATION_RATE)

        # Select the best individuals for mutation
        best_fitted_individuals = utils.get_index_of_the_best_fitted_individuals(self.population_score,
                                                                                 num_individuals_to_mutate)

        for index in best_fitted_individuals:
            individual_to_mutate = self.population[index]

            # Identify the worst sub-board based on the selected evaluation mode
            score, worst_fitted_range = utils.calculate_worst_fitted_sub_board(individual_to_mutate, self.mode)
            from_row, to_row, from_column, to_column = worst_fitted_range

            # Extract the selected rows for mutation
            row_genes = individual_to_mutate[from_row:to_row]
            sub_board = []

            # Ensure sub-board size meets the RL agent's required dimensions
            while len(row_genes) < config.AGENT_WINDOW_ROW:
                row_genes.append([5] * config.AGENT_WINDOW_COLUMN)  # Fill missing rows with gaps (5)

            # Extract the sub-region from each row and ensure it meets RL input requirements
            for genes in row_genes:
                sub_genes = genes[from_column:to_column]
                while len(sub_genes) < config.AGENT_WINDOW_COLUMN:
                    sub_genes.append(5)  # Fill missing columns with gaps (5)
                sub_board.append(sub_genes)

            # Perform mutation using the RL agent
            env = Environment(sub_board, convert_data=False)
            agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
            agent.load(model_path)
            state = env.reset()

            while True:
                action = agent.predict(state)
                _, next_state, done = env.step(action)
                state = next_state
                if done == 0:
                    break

            env.padding()

            # Replace the mutated sub-board within the original individual
            for idx, sequence in enumerate(env.aligned):
                if idx < len(row_genes):  # Ensure we do not modify extra padding rows
                    row_genes[idx][from_column:to_column] = sequence

            individual_to_mutate[from_row:to_row] = row_genes

    def print_hall_of_fame(self):
        """
        Prints the best individual currently stored in the Hall of Fame.

        - Displays the nucleotide sequence instead of numerical representation.
        - Shows the corresponding fitness score based on the GA mode.
        - Ensures visibility of the best solution found so far.

        Returns:
            None
        """
        if not self.hall_of_fame:
            print("⚠️ Hall of Fame is empty! No best individual found yet.")
            return

        hof_individual, hof_score = self.hall_of_fame

        print("\n🏆 HALL OF FAME - BEST INDIVIDUAL SO FAR:")
        print("=" * 50)

        # Convert numerical sequence to nucleotides
        converted_hof = utils.get_nucleotides_seqs(hof_individual)

        for sequence in converted_hof:
            print(f"   🧬 {sequence}")

        # Display fitness scores based on mode
        if self.mode in {"sp", "cs"}:
            score_type = "SP" if self.mode == "sp" else "CS"
            print(f"   🎯 Best {score_type} Score: {hof_score[1]}")
        elif self.mode == "mo":
            print(f"   🎯 SP Score: {hof_score[1]}")
            print(f"   🎯 CS Score: {hof_score[2]}")

        print("=" * 50)

    def print_population(self):
        """
        Prints the current population along with their fitness scores.
        Provides a detailed view of the sequences and their fitness values.
        """
        print("\n📌 CURRENT POPULATION (Iteration {}):".format(self.current_iteration))
        print("=" * 50)

        if not self.population:
            print("⚠️ Empty Population! Make sure your population was generated correctly.")
            return

        for i, (individual, scores) in enumerate(zip(self.population, self.population_score), start=1):
            print(f"🔹 Individual {i}:")
            converted_individual = utils.get_nucleotides_seqs(individual)

            for sequence in converted_individual:
                print(f"   🧬 {sequence}")  # Print individual's sequences

            # Extract fitness scores based on mode
            if self.mode in {'sp', 'cs'}:
                score = scores[1]  # Single score (SP or CS)
                score_type = "SP" if self.mode == "sp" else "CS"
                print(f"   🎯 Fitness Score ({score_type}): {score}")

            elif self.mode == 'mo':
                sp_score, cs_score = scores[1], scores[2]
                print(f"   🎯 SP Score: {sp_score}")
                print(f"   🎯 CS Score: {cs_score}")

            print("-" * 50)

        self.print_hall_of_fame()
        print("\n✅ Population entirely printed.\n")

    def run(self, model_path, debug_mode=False):
        """
        Runs the Genetic Algorithm (GA) pipeline for multiple iterations using the selected fitness mode.

        Args:
            model_path (str): Path to the reinforcement learning (RL) model used for mutation.
            debug_mode (bool, optional): If True, enables detailed logging of each step. Defaults to False.

        Returns:
            list: The best individual's aligned sequences.
        """

        if debug_mode:
            print("🔵 Step 1: Initializing population...")

        self.generate_population()

        if debug_mode:
            print("✅ Population initialized!")

        self.calculate_fitness_score()

        if debug_mode:
            self.print_population()

        # Validate RL model window size
        first_individual = self.population[0]
        if config.AGENT_WINDOW_ROW > len(first_individual):
            raise ValueError("❌ AGENT_WINDOW_ROW is larger than the sequence count.")

        sequence_min_length = min(len(genes) for genes in first_individual)
        if config.AGENT_WINDOW_COLUMN > sequence_min_length:
            raise ValueError("❌ AGENT_WINDOW_COLUMN is larger than the shortest sequence length.")

        if debug_mode:
            print("🟢 Step 2: Starting Genetic Algorithm iterations...")

        # GA Iterations
        for i in range(config.GA_ITERATIONS):
            if debug_mode:
                print(f"🔄 Iteration {i + 1}/{config.GA_ITERATIONS}...")

            self.current_iteration += 1

            if debug_mode:
                print("  🔹 Applying mutation...")

            self.mutation(model_path)

            if debug_mode:
                print("  ✅ Mutation applied.")
                print("  🔹 Calculating fitness score...")

            self.calculate_fitness_score()

            if debug_mode:
                print("  ✅ Fitness score calculated.")
                self.print_population()

                print("  🔹 Performing selection...")

            self.selection()

            if debug_mode:
                print("  ✅ Selection completed.")
                self.print_population()

                print("  🔹 Performing horizontal crossover...")

            self.horizontal_crossover()

            if debug_mode:
                print("  ✅ Horizontal crossover completed.")
                self.print_population()

        if debug_mode:
            print("🟠 Step 3: Selecting the best chromosome...")

        best_chromosome, _ = self.hall_of_fame

        if debug_mode:
            print("✅ Best chromosome selected.")
            print("📜 Extracting aligned sequences...")

        aligned_seqs = utils.get_nucleotides_seqs(best_chromosome)

        if debug_mode:
            print("✅ Alignment completed!")

        return aligned_seqs

