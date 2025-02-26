# Genetic Algorithm and Deep reinforcement learning for MSA (GA-DPAMSA)
GA-DPAMSA is an application for multiple sequence alignment using a Deep Reinforcement Learning model (DPAMSA) together with a genetic algorithm. Starting from the [DPAMSA model](https://github.com/ZhangLab312/DPAMSA), a genetic algorithm was built on it, which allows for higher performance than the base model and allows the use of a Reinforcement learning agent trained for a problem $P$, to also be used for a problem $P_2$ where $P_2$ is more complex (has larger dimensions in terms of the number of sequences and bases per sequence), without the need to re-train the model, while still achieving excellent performance. This is achieved by making sure that in the mutation phase of the genetic algorithm, the reinforcement learning agent is employed to operate on the problem that has the same dimension as the problem $P$, i.e., the one on which it was initially trained.

---

# Table of contents
- [Dataset creation](#dataset-creation)

- [How the genetic algorithm is implemented](#how-the-genetic-algorithm-is-implemented-)

- [How to use GA-DPAMSA?](#how-to-use-ga-dpamsa)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Inference](#inference)
- [Benchmarking](#benchmarking)

---

## Dataset creation
Running the script ```generate_dataset.py``` is possible to create a stable and controlled dataset for the experiments.

```py
# ================= CONFIGURATION ================= #
# User-configurable parameters
num_sequences = 6            # Number of DNA sequences per dataset
sequence_length = 30         # Length of each sequence
mutation_rate = 0.10         # Mutation probability (10%)
gap_rate = 0.05              # Gap insertion probability (5%)
number_of_datasets = 50      # Total number of datasets to generate
min_score_threshold = 10     # Minimum alignment score threshold
max_score_threshold = None   # Maximum alignment score threshold (None = no limit)

# Conserved block settings
num_conserved_blocks = 1      # Number of conserved blocks per sequence
conserved_block_sizes = [10]  # List of block sizes (one size per block)

# Additional options
fixed_block_position = False  # True = fixed position, False = random position
mutate_inside_blocks = False  # True = mutations inside blocks allowed, False = only outside
```

The parameters in this script are carefully tuned to balance variability and conservation in the generated DNA sequences. The number of sequences and their lengths directly determine the complexity of the dataset; more sequences and longer lengths allow for more sites where mutations or gap insertions can occur, making the alignment more challenging. The mutation rate introduces diversity by randomly altering bases, while the gap rate injects missing information into the sequences by replacing bases with gap characters. Higher values for either result in greater divergence among the sequences, potentially lowering alignment scores.

Conserved blocks are integrated into each sequence to simulate biologically important regions that remain relatively unchanged. The number and size of these blocks, along with their positioning—whether fixed or random—ensure that a portion of each sequence retains high similarity. The option to prevent mutations within these blocks further reinforces their conservation, which is essential for certain alignment scenarios. Finally, alignment score thresholds filter the generated datasets to maintain a desired level of overall similarity.

Together, these parameters interact to create datasets that can be tailored for different benchmarking or training purposes, striking a balance between variability and conservation in a controlled manner.

---

## How the genetic algorithm is implemented 
The goal is to make sure that given 4 sequences to align, each of length 8, like the one in the following figure (so a $4x8$ board

![Board](/img/board.png)

through the use of a genetic algorithm, to use an RL agent to align the board, but having an agent that has been trained on a $2x4$ board, so it would basically only know how to solve the problem on a $sub-board$, i.e., the one in the green square in the following figure

![SubBoard](/img/subboard.png)

In the following sections, we will detail how the genetic algorithm was created to solve this problem, explaining how the operations of *population generation*, *fitness score*, *selection*, *crossover*, and *mutation* were implemented.

### Population generation
Given sequences to align, a matrix is created. To apply RL, each nucleotide is mapped to a number according to the following rule:

```
A:1, T:2, C:3, G:4, a:1, t:2, c:3, g:4, -:5
```
Note that the GAP is encoded with $5$. Thus, after this phase, we will have a $4x8$ matrix of integers. At this point, the population of the genetic algorithm is generated, where each individual will esentially be a matrix (from now on, we will also refer to the matrix as $the\ board$). In the following figure, you can see what a population of 4 individuals looks like for the alignment of 4 sequences, each having 8 bases.

![Population](/img/population.png)

The number of individuals will then be configurable by the user based on their needs (in the [config.py](config.py) file ).

### Fitness score
The genetic algorithm has three modes:
1.  **Maximize Sum of pairs** : The fitness score is based on the sum of pairs.
2.  **Maximize Column score** : The fitness score is based on the column score.
3.  **Maximize intersection** : The fitness score calculates both column score and sum of pairs, then are selected best individuals based on the intersetion between the two metrics.
We do not calculate the fitness score immediately after generating the population because, in the initial phase, the population consists of identical individuals; the same board is replicated $n$ times (where $n$ can be customized by the user). We introduce the fitness score now because it is useful for understanding how the mutation phase will operate. The fitness score essentially involves calculating the sum-of-pairs, or column score or both, for each individual in the population.
Below is the formula used to calculate sum-of-pairs:

![Sum-of-pairs](/img/sum-of-pairs-formula.png)

Below is the formula used to calculate column score:

![Sum-of-pairs](/img/colum_score_formula.png)

 It is important to note that if the algorithm is used with an RL model that employs different weights for gaps, matches, and mismatches, the fitness score calculation must also be adjusted to use the same weights. This avoids inconsistencies between the weights used by the RL agent and those used by the algorithm. These values can be easily modified in the [config.py](config.py) file of the application.

 ### Mutation
The mutation phase is a critical component of the GA-DPAMSA pipeline. Its goal is to improve the quality of candidate alignments by selectively altering their weakest regions. This phase uses a reinforcement learning (RL) agent to target and modify the sub-region (sub-board) that performs worst according to specific fitness metrics.

**Selection of Individuals for Mutation**

At the beginning of the mutation phase, the algorithm determines how many individuals should undergo mutation. This number is computed as a fraction of the total population, based on a preset mutation rate in the [config.py](config.py). The best-fitted individuals (those with the highest fitness scores) are selected for mutation. By focusing on high-quality candidates, the algorithm aims to improve only the regions that are most likely to benefit from targeted changes.

**Identification of the Worst-Fitted Sub-Board**

For each selected candidate alignment, the mutation phase identifies the worst-performing sub-board. This is achieved by evaluating various sub-regions using metrics such as the Sum-of-Pairs (SP) score and/or the Column Score (CS). The sub-board with the lowest score is considered the weakest link in the alignment and is chosen as the target for mutation.

**Sub-Board Extraction and Preprocessing**

Once the worst sub-board is identified, the following steps are performed:
- **Extraction:** The specific rows and columns corresponding to the poor-performing region are extracted from the candidate alignment.
- **Preprocessing:** The extracted sub-board is adjusted to meet the input requirements of the RL agent. This includes:
  - Padding the sub-board with gap characters (if necessary) so that it matches the required number of rows.
  - Ensuring that each row of the sub-board contains the required number of columns by appending gaps as needed.

This preprocessing is essential to ensure the RL agent receives input data of consistent dimensions.

**RL Agent Mutation Process**
With the sub-board prepared, the mutation process proceeds as follows:
1. **Environment Setup:**  
   An environment is instantiated with the preprocessed sub-board. This environment encapsulates the state of the sub-board and sets up parameters such as the available actions for mutation.
   
2. **RL Agent Initialization:**  
   A Deep Q-Network (DQN) agent is initialized and loaded with a pre-trained model specified by the provided model path. The agent is responsible for determining which mutations to apply.
   
3. **Iterative Mutation:**  
   The agent interacts with the environment in a loop:
   - It predicts an action based on the current state of the sub-board.
   - The predicted action is applied, leading to a modified sub-board and a new state.
   - This loop continues until a termination condition is met (i.e., when a specific "done" flag indicates that the mutation process should stop).
   
4. **Post-Mutation Adjustment:**  
   Once the mutation loop concludes, the mutated sub-board is passed through a padding function to ensure that any changes are correctly integrated and that the sub-board’s dimensions remain consistent.

![Mutation](/img/mutation-dpamsa.png)

**Integration of Mutated Sub-Board into the Candidate Alignment**
After mutation, the algorithm reintegrates the modified sub-board back into the original candidate alignment. The region corresponding to the worst-performing sub-board is replaced with its mutated version. This targeted approach helps improve the overall fitness of the candidate alignment while leaving the well-aligned regions untouched.

### Selection
In our Genetic Algorithm, selection is carried out using an elitist strategy that is governed by the selection rate defined in [config.py](config.py). Essentially, only the top-performing candidates—determined by their fitness scores—are allowed to survive into the next generation. The method of evaluating fitness depends on the mode of operation. In single-objective modes (either Sum-of-Pairs or Column Score), candidates are ranked directly on that one metric. In multi-objective mode, both metrics are considered together, often through a Pareto Front analysis. This elitism ensures that only a fixed fraction of the best candidates, as dictated by the selection rate, are retained to form the basis for the next generation’s evolution.

### Crossover
The crossover phase is implemented using a horizontal crossover strategy. During this phase, new candidate alignments are generated by combining parts from two parent candidates. Specifically, two distinct parents are randomly selected from the current population, and a random crossover point is chosen along the rows of their alignments. The new candidate is then created by taking the top portion (up to the crossover point) from one parent and the bottom portion (from the crossover point onward) from the other.

Deep copies of the parent segments are used to ensure that the newly generated candidate is completely independent of its parents. This prevents any unintended modifications in the offspring due to subsequent mutations in the parent alignments. The crossover process is repeated until the number of new individuals, when added to the existing population, reaches the target population size defined in the configuration file.

After the offspring are generated, the fitness scores of the entire population are recalculated. This update reflects the changes introduced by the crossover, ensuring that only the most promising candidates are available for the next phase of the algorithm.
![Horizontal-crossover](/img/horizontal-crossover.png)


### Algorithm flow
Once the application is started, the following operations will be executed over a specified number $n$ of iterations (where $n$ the number of iterations configured for the algorithm, see section [configuration](#configuration)):

The overall flow of the GA-DPAMSA algorithm can be summarized as follows:

- **Initial Population Generation:**
  - Create candidate alignments by converting input sequences to numerical representations.
  - Use a mix of:
    - **Exact Copies:** A fraction of the population (defined by the clone rate) is generated as direct copies.
    - **Modified Individuals:** The rest of the population is created by inserting random gaps into the sequences (based on the gap rate).

- **Fitness Evaluation and Cleanup:**
  - For each candidate alignment:
    - Pad sequences to ensure equal length.
    - Clean unnecessary gap columns.
    - Calculate fitness scores using:
      - **Single-Objective Metrics:** Sum-of-Pairs (SP) or Column Score (CS).
      - **Multi-Objective Metrics:** A combination of SP and CS (often via Pareto Front analysis).
  - Update the Hall of Fame with the best candidate seen so far.

- **Evolutionary Iterations (Generations):**
  - **Mutation Phase:**
    - Determine the number of individuals to mutate based on the mutation rate.
    - For each selected candidate:
      - Identify the worst-performing sub-region (sub-board) using fitness metrics.
      - Use a Reinforcement Learning (RL) agent to iteratively mutate this sub-board.
      - Replace the original sub-board with the mutated version.
      
  - **Selection Phase:**
    - Apply an elitist strategy by retaining the top fraction of candidates (as defined by the selection rate in the config).
    - In single-objective modes, rank candidates directly on SP or CS scores.
    - In multi-objective mode, consider both metrics (using Pareto Front analysis if needed).

  - **Crossover Phase:**
    - Generate new candidate alignments through horizontal crossover:
      - Randomly select two distinct parent candidates.
      - Choose a random crossover point along the rows.
      - Create offspring by combining the top portion of one parent with the bottom portion of the other.
      - Use deep copies to ensure the offspring are independent of their parents.
      
  - **Population Update:**
    - Replace the current population with the selected candidates and newly generated offspring.
    - Recalculate fitness scores for the updated population.
    - Update the Hall of Fame if a better candidate is found.

- **Final Output:**
  - After the specified number of iterations, extract the best alignment stored in the Hall of Fame.
  - This alignment represents the final, refined solution produced by the GA-DPAMSA process.

---

# How to use GA-DPAMSA
First clone the repository locally
```sh
git clone https://github.com/FLaTNNBio/GA-DPAMSA
```
Place in the project directory and create a virtual environment (highly recommended)
```sh
python3 -m venv venv
```
Install all required libraries with the following command
```sh
pip install -r requirements.txt
```
Then appropriately configure the [config.py](./config.py) file according to your needs ([see next section for info](#configuration))

---

## Configuration
Several parameters can be set from the [config.py](./config.py) file.

```py
# ===========================
# Genetic Algorithm (GA) Parameters
# ===========================
GAP_PENALTY = -4  # Penalty for inserting a gap
MISMATCH_PENALTY = -4  # Penalty for a mismatch
MATCH_REWARD = 4  # Reward for a correct match

AGENT_WINDOW_ROW = 3  # Number of rows in the agent's observation window
AGENT_WINDOW_COLUMN = 30  # Number of columns in the observation window
GA_ITERATIONS = 3  # Number of iterations for genetic evolution
POPULATION_SIZE = 5  # Population size for genetic algorithm
CLONE_RATE = 0.25  # % of the population to be an exact copy of the input sequences during Population Generation Phase
GAP_RATE = 0.05  # % of Gap to be added to an individual during Population Generation Phase (calculated on seq. length)
SELECTION_RATE = 0.5  # % of the population to be selected following a certain criteria
MUTATION_RATE = 0.25  # % of the population undergo mutation
```
As you can see it is possible to change the weight that is given to a gap, mismatch, match in calculating the sum of pairs through changing the values respectively: ```GAP_PENALTY```, ```MISMATCH_PENALTY``` and ```MATCH_REWARD```. In case they are changed, the model needs to be retrained, as the same values are also used to calculate the reward for the RL agent.
Then it is possible to change the parameters of the genetic algorithm, such as the number of individuals in the population (```POPULATION_SIZE```), the number of iterations after which to stop the algorithm (```GA_ITERATIONS```), etc.

The  ```AGENT_WINDOW_ROW``` and ```AGENT_WINDOW_COLUMN``` parameters should be set to the same value as the dataset used to run the DPAMSA model training, so if for example the RL model was trained on dataset containing 3 sequences to be aligned, where each sequence has 30 bases, the values should be set as in the example above.

It is also possible to modify other parameters relating to the training of the DPAMSA model from the [config.py](./config.py), for those see [DPAMSA  README.md](./DPAMSA/README.md).

---

## Training

Check [DPAMSA  README.md](./DPAMSA/README.md).

---

## Inference

**Prerequisites**
- **Dataset Module:** By default, the script uses the dataset from `datasets.inference_dataset.dataset1_6x60bp`. Make sure this module is available or modify the `DATASET` constant as needed.
- **Trained RL Model:** A trained reinforcement learning (RL) model is required for the mutation phase. The default model is specified by `INFERENCE_MODEL` (e.g., `'model_3x30'`).
- **Configuration File:** The `config.py` file should include parameters like population size, number of GA iterations, selection rate, mutation rate, and window sizes.

**Default Settings**

At the top of the script, several constants control the behavior of the inference:

- **GA_MODE:**  
  Specifies the evaluation mode. Options include:  
  - `'sp'` for Sum-of-Pairs mode  
  - `'cs'` for Column Score mode  
  - `'mo'` for Multi-Objective mode

- **DATASET:**  
  Points to the dataset module containing the sequences. The default is `datasets.inference_dataset.dataset1_6x60bp`.

- **INFERENCE_MODEL:**  
  The identifier or path to the trained RL model used during mutation (default: `'model_3x30'`).

- **DEBUG_MODE:**  
  A boolean flag that enables detailed logging if set to `True` (default is `False`).

Feel free to modify these values to suit your inference needs.

**Running the Script**

1. **Direct Execution:**
   - Open a terminal or command prompt.
   - Navigate to the directory containing `mainGA.py`.
   - Run the script with the command:
     ```bash
     python mainGA.py
     ```
   - The script will iterate over the datasets, process each one, and output progress information via a progress bar.

2. **Customizing Inference Parameters:**
   - Open `mainGA.py` in your preferred text editor.
   - Modify the constants at the top of the file to change:
     - The inference mode (`GA_MODE`), for example, set it to `'sp'` for Sum-of-Pairs.
     - The dataset module (`DATASET`) if you want to use a different dataset.
     - The RL model path (`INFERENCE_MODEL`) if you have a different model.
     - The debug mode (`DEBUG_MODE`) if you need detailed logging.
   - Save your changes and run the script as described above.

**Output Files**

After execution, the script generates two key output files:
- **Report File:**  
  Contains detailed information for each dataset processed, including alignment length, fitness scores, and the final alignment.
  
- **CSV File:**  
  Summarizes key metrics for each dataset, making it easy to review the overall performance of the GA-DPAMSA inference.

The paths to these files are constructed using settings from `config.py` and are printed to the console upon completion.

**Troubleshooting**

- **Parameter Errors:**  
  If you encounter errors regarding window sizes or sequence lengths, double-check that the values in `config.py` (e.g., `AGENT_WINDOW_ROW` and `AGENT_WINDOW_COLUMN`) match the dimensions of your dataset.

---

## Benchmarking

The benchmarking framework evaluates **multiple sequence alignment (MSA) methods** by comparing the performance of **GA-DPAMSA**, the **DPAMSA deep reinforcement learning model**, and **other external MSA tools**, by following these key steps:  

1. **Dataset Selection** – Choosing an MSA dataset for benchmarking.  
2. **Running GA-DPAMSA** – Performing sequence alignment using the Genetic Algorithm with Reinforcement Learning.  
3. **Running DPAMSA** – Evaluating the standalone deep reinforcement learning model (DPAMSA).  
4. **Running External MSA Tools** – Comparing GA-DPAMSA against traditional MSA tools.  
5. **Generating Reports & Plots** – Summarizing results and visualizing performance metrics.

### **Tools Used**

GA-DPAMSA and DPAMSA were benchmarked against standard bioinformatics MSA tools:  

| **Tool**             | **Description** |
|----------------------|---------------|
| **[ClustalOmega](http://www.clustal.org/omega/)** | A fast, accurate multiple sequence alignment tool. |
| **[MSAProbs](https://msaprobs.sourceforge.net/homepage.htm#latest)**     | Uses probabilistic consistency-based alignment. |
| **[ClustalW](http://www.clustal.org/clustal2/)**     | A widely used alignment tool with progressive alignment. |
| **[MAFFT](https://mafft.cbrc.jp/alignment/server/index.html)**        | High-speed multiple sequence alignment tool using FFT. |
| **[MUSCLE5](https://www.drive5.com/muscle/)**      | Iterative alignment tool designed for high accuracy. |
| **[UPP](https://github.com/smirarab/sepp/blob/master/README.UPP.md)**          | Handles large-scale alignments using progressive refinement. |
| **[PASTA](https://github.com/smirarab/pasta)**        | A highly scalable multiple sequence alignment algorithm. |

Each of these tools follows different alignment strategies, providing a **diverse benchmark** for evaluating GA-DPAMSA.

**Note**: the majority of these tools only works on Linux.


### How Benchmarking Works  

1. **Running the Benchmarking Script**

    Execute the benchmarking process using:
    ```sh
     python run_benchmarks.py
    ```

2. **Selecting Benchmarking Options**

    The script prompts the user to select which tools to benchmark:
    
    - Option 1: Run GA-DPAMSA and DPAMSA.
    - Option 2: Run external MSA tools only.
    - Option 3: Run both GA-DPAMSA/DPAMSA and external tools.

3. **Dataset Handling**

    Datasets are stored in the `datasets/` folder.

    The script processes datasets sequentially and runs each tool on the same dataset.

4. **Evaluation Metrics**

    Each method is evaluated using the following alignment quality metrics:

    | **Metric**            | **Description** |
    |---------------------|---------------|
    |Sum-of-Pairs (SP) | Measures the total pairwise alignment similarity. Higher values indicate better alignments.
    |Column Score (CS) |Calculates how many columns are fully matched. Expressed as a percentage.
    |Alignment Length (AL) |The total length of the aligned sequences after insertion of gaps.
    |Exact Matches (EM) | The number of columns where all sequences perfectly match.

### Output

After running the benchmarking script, results are stored in CSV files and text reports.

- **Reports Directory**
    
    Results are saved in:
    ```bash
    results/reports/GA-DPAMSA/
    ```
    
    Each dataset will have a corresponding report file, named as:

    ```bash
    <dataset_name>_MO.txt
    ```
    
    Example content of a report:
    ```mathematica
    File: test5
    Number of Sequences (QTY): 3
    Alignment Length (AL): 30
    Sum of Pairs (SP): -104
    Exact Matches (EM): 5
    Column Score (CS): 0.167
    Alignment:
    AGTCTCGACGCGACCACCGATTATGACATT
    CCTTTGGCCATCAGGAAAGTAGGCCGCATA
    TGTACCGTATCCGGCAGCGCAGATCCCCCG
    ```

- **CSV Results**

    Performance metrics for all tools are stored in:

    ```bash
    results/benchmarks/csv/
    ```

    Each tool has its own CSV file with the following structure:
    
    ```bash
    Dataset Name	QTY	AL	SP	EM	CS
    dataset1	6	72	142	50	0.69
    dataset2	3	35	89	20	0.74
    ```

- **Performance Plots**

    Once benchmarking is complete, the script generates comparison plots, stored in:
    
    ```bash
    results/benchmarks/charts/
    ```
   
    These visualizations include:
   - Box Plots – to show SP and CS distributions.
   - Bar Plots – for average SP and CS values.

---


