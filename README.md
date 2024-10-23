# Genetic Algorithm and Deep reinforcement learning for MSA (GA-DPAMSA)
GA-DPAMSA is an application for multiple sequence alignment using a Deep Reinforcement Learning model (DPAMSA) together with a genetic algorithm. Starting from the [DPAMSA model](https://github.com/ZhangLab312/DPAMSA), a genetic algorithm was built on it, which allows for higher performance than the base model and allows the use of a Reinforcement learning agent trained for a problem $P$, to also be used for a problem $P_2$ where $P_2$ is more complex (has larger dimensions in terms of the number of sequences and bases per sequence), without the need to re-train the model, while still achieving excellent performance. This is achieved by making sure that in the mutation phase of the genetic algorithm, the reinforcement learning agent is employed to operate on the problem that has the same dimension as the problem $P$, i.e., the one on which it was initially trained.

# Table of contents
- [GA-DPAMSA](#genetic-algorithm-and-deep-reinforcement-learning-for-msa-ga-dpamsa)

- [How the genetic algorithm is implemented](#how-the-genetic-algorithm-is-implemented)

- [How to use GA-DPAMSA?](#how-to-use-ga-dpamsa)
    - [Configuration](#configuration)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
    - [Training](#training)
- [Results](#result)

- [How change the RL Model](#how-change-the-rl-model)

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

The number of individuals will then be configurable by the user based on their needs (in the [config.py](config.py) file ). In this generation phase, all possible ranges of size $2x4$ on the board are also calculated (specifically, those ranges for which the agent is trained). Each range is calculated to ensure that all ranges are different and there is no overlap, to prevent the RL agent from operating in areas where it has already performed an alignment on another individual.

### Fitness score
We do not calculate the fitness score immediately after generating the population because, in the initial phase, the population consists of identical individuals; the same board is replicated $n$ times (where $n$ can be customized by the user). We introduce the fitness score now because it is useful for understanding how the mutation phase will operate. The fitness score essentially involves calculating the sum-of-pairs for each individual in the population. Below is the formula used to calculate sum-of-pairs:

![Sum-of-pairs](/img/sum-of-pairs-formula.png)


 It is important to note that if the algorithm is used with an RL model that employs different weights for gaps, matches, and mismatches, the fitness score calculation must also be adjusted to use the same weights. This avoids inconsistencies between the weights used by the RL agent and those used by the algorithm. These values can be easily modified in the [config.py](config.py) file of the application.

 ### Mutation
 After generating the population, we proceed with the mutation operation, where the RL algorithm comes into play. We have provided two different types of mutations to evaluate their performance and give the user the freedom to choose the most suitable one based on the use case. The first is the random mutation, while the second is the mutation on the best fitted individuals.

 1. **Random mutation** : The random mutation involves randomly selecting an individual in the population to mutate. Based on the number of possible different sub-boards we have, we perform the same number of mutations. For example, in the case of a $4x8$ board like the one in the figure, and an RL agent trained to operate on $2x4$ boards, we generate all unique ranges on the $4x8$ board (in this example, there are 4). For each range, we randomly select an individual in the population and apply the RL agent to that individual in one of those ranges (the range to operate on is also chosen randomly). After performing the mutation, that range is eliminated, and we proceed to mutate another individual on a different sub-board. The mutation phase will end when we have exhausted all possible unique ranges of the sub-boards.

 2. **Mutation on the best fitted individuals**: In this case, instead of randomly selecting the individual to operate on, we choose the one with the highest sum-of-pairs score. Specifically, before performing the mutation, the sum-of-pairs is calculated for all individuals. At this point, the best $n$ individuals are selected for mutation (those with the highest sum-of-pairs scores). The number $n$ of individuals to mutate is not based on the number of different sub-boards, as in the previous case, but is a certain percentage of individuals in the population (customizable by the user in the [config.py](config.py) file). For each of these $n$ selected individuals with the highest sum-of-pairs, the sub-board where the RL agent will operate is the one with the lowest sum-of-pairs. In other words, during this mutation phase, given an individual (selected because it had the highest sum-of-pairs on the entire board), the sum-of-pairs is also calculated for each unique sub-board, and the sub-board with the lowest sum-of-pairs is where the RL agent will operate to try to improve that solution. This mutation is computationally more expensive but, as we will see in the [results section](#results), it brings significant benefits, generally leading to better sequence alignment (in terms of sum-of-pairs). The objective of providing these two different methods is to see the extent of improvement in the second case compared to the first, but also to give the user the freedom to prioritize either faster alignment computation (using the first method) or higher alignment quality at the expense of performance. In the following figure, we can see the RL agent operating on the individual, performing the mutation.

![Mutation](/img/mutation-dpamsa.png)

### Selection
The selection phase of our algorithm involves calculating the sum-of-pairs for each individual and ordering them based on the obtained value. At this point, only the top $n$ individuals with the highest sum-of-pairs will be selected to generate new individuals and proceed to the next iteration. The value of $n$ can be customized by the user based on their needs and how the algorithm responds to their problem.

### Crossover
For the crossover phase, we have provided two different methods for generating new individuals: **vertical crossover** and **horizontal crossover**.  For both methods, two individuals (from those that passed the selection phase) are chosen as pairs, referred to as *Parent 1* and *Parent 2*. From each pair, a new individual will be created until we reach the number 
$n$ of individuals that we set at the beginning of the algorithm. The selection of *Parent 1* and *Parent 2* is random among the individuals that passed the selection phase.
- **Vertical crossover**: In vertical crossover, *Parent 1* and *Parent 2* are split in half vertically (the point of division is determined by averaging the length of each row and cutting exactly at the average length). To generate the new individual, the first half is taken from Parent 1 and the second half from Parent 2. The choice of *Parent 1* and *Parent 2* is made randomly.
In the figure below, an example of this operation is shown.

![Vertical-crossover](/img/vertical-crossover.png)

- **Horizontal crossover**: In horizontal crossover, the individual is split along the rows. Specifically, the number of rows of the board is counted, and a division is made in the middle of the board along the rows (if odd, Parent 1 will contribute one more row than Parent 2). At this point, we combine by taking the first half from Parent 1 and the second half from Parent 2. The figure below shows an example of this crossover.

![Horizontal-crossover](/img/horizontal-crossover.png)


### Algorithm flow
Once the application is started, the following operations will be executed over a specified number $n$ of iterations (where $n$ the number of iterations configured for the algorithm, see section [configuration](#configuration)):

1. Mutation: Mutate the individuals in the population.

2. Calculate Fitness Score: Compute the fitness score for each individual.

3. Selection: Select the best individuals to pass to the next generation for generating new individuals.

4. Crossover: Create new individuals through crossover operations.

At the end of $n$ iterations, the fitness score is recalculated for each individual (since the cycle ends with crossover, we evaluate the fitness of the newly generated population from the last iteration).
Next, the individual with the highest sum-of-pairs score is extracted from the population. Its board is then converted back from integers to characters (representing nucleotides). This aligned sequence, along with its sum-of-pairs value, is displayed on the screen and saved into a file in the [results](./result/reportDPAMSA_GA/) folder of the project.

# How to use GA-DPAMSA
First clone the repository locally
```sh
git clone https://github.com/strumenti-formali-per-la-bioinformatica/GA-DPAMSA.git
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

## Configuration
Several parameters can be set from the [config.py](./config.py) file.

```py
GAP_PENALTY = -4
MISMATCH_PENALTY = -4
MATCH_REWARD = 4
GA_POPULATION_SIZE = 5
GA_NUM_ITERATION = 3
GA_NUM_MOST_FIT_FOR_ITER = 2
GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER = 0.20 #20%

#This depend from the training dataset given to the DQN
AGENT_WINDOW_ROW = 3
AGENT_WINDOW_COLUMN = 30

DATASET_ROW = 6
DATASET_COLUMN = 60

```
As you can see it is possible to change the weight that is given to a gap, mismatch, match in calculating the sum of pairs through changing the values respectively: ```GAP_PENALTY```, ```MISMATCH_PENALTY``` and ```MATCH_REWARD```. In case they are changed, the model needs to be retrained, as the same values are also used to calculate the reward for the RL agent.
Then it is possible to change the parameters of the genetic algorithm, such as the number of individuals in the population (```GA_POPULATION_SIZE```), the number of iterations after which to stop the algorithm (```GA_NUM_ITERATION```), the number of individuals to be propagated at each iteration (```GA_NUM_MOST_FIT_FOR_ITER```), and the percentage of individuals to be mutated at each iteration, only in the case best-fitted-mutation is chosen (```GA_PERCENTAGE_INDIVIDUALS_TO_MUTATE_FOR_ITER```).
Depending on the dataset that is used for model inference and the training dataset used for DPAMSA, the following parameters need to be changed: ```AGENT_WINDOW_ROW```, ```AGENT_WINDOW_COLUMN```, ```DATASET_ROW```, ```DATASET_COLUMN```.
The  ```AGENT_WINDOW_ROW``` and ```AGENT_WINDOW_COLUMN``` parameters should be set to the same value as the dataset used to run the DPAMSA model training, so if for example the RL model was trained on dataset containing 3 sequences to be aligned, where each sequence has 30 bases, the values should be set as in the example above.
Instead, the ```DATASET_ROW``` and ```DATASET_COLUMN``` parameters must be set based on the dataset we are passing to GA-DPAMSA to perform the inference. So if the dataset on which we want to make inference contains tests where 6 sequences need to be aligned where each sequence contains 60 bases, the values ​​must be configured as in the example above. It is also possible to modify other parameters relating to the training of the DPAMSA model from the [config.py](./config.py), for those see [DPAMSA reference paper](https://academic.oup.com/bioinformatics/article/39/11/btad636/7323576).

## Inference
With the following command you can perform inference on a dataset with GA-DPAMSA:
```py
python3 mainGA.py
```
To change dataset for inference, you need to insert it into the [datasets/inference_dataset](./datasets/inference_dataset/) folder and import it into the [mainGA.py](./mainGA.py) file, essentially you need to modify the first line, indicating the name of the dataset.
From the [mainGA.py](./mainGA.py) file it is possible to change the type of mutation and the type of crossover of the genetic algorithm.

```python
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

```
In particular, in the ```for``` executed based on the iteration number of the genetic algorithm, it is possible to remove or insert the comment to the function call. In the example above we see that ```ga.mutation_on_best_fitted_individuals_worst_sub_board(model_path)``` function is called, you can comment this and uncomment the ```ga.random_mutation(model_path)``` function to use the random mutation. As we can see, the same goes for the horizontal or vertical crossover with the functions ```ga.horizontal_crossover``` and ```ga.vertical_crossover()```
If you have trained a new model with DPAMSA and you want to use that for inference, just edit the inference function call in mainGA.py and insert the path to the new model.

## Evaluation
If you want to test GA-DPAMSA, you can run some scripts that run the benchmark on already prepared datasets or you can create some new datasets made from synthetic sequences, launching the [create_fataset.py](./datasets/create_dataset.py) script in the datasets folder. From this script you can personalize the  ```num_sequences```, the ```sequence_length```, the ```mutation_rate```, the ```number_of_dataset``` and the ```DATASET_NAME```. The output dataset will be placed in the [/datasets](./datasets/) folder.
Useful fasta files will also be created to give them as input to ClustalOmega, ClustalW and MSAProbs in the [/datasets/fatsta_files](./datasets/fasta_files/) directory.
By running the [mainGA.py](./mainGA.py) file on one of these datasets, all the results are inserted in the [reportDPAMSA_GA folder](./result/reportDPAMSA_GA/), with the value of the sum-of-pairs and the alignment, for each of the tests in the dataset.
If you want to compare the results from GA-DPAMSA with ClustalW, ClustalOmega and MSAProbs, you can execute the [execute_benchmark.py](./execute_benchmark.py) script. This script will automatically execute the 3 tools and give as input the fasta files of the desired dataset (configurable within the script, giving it the path of the folder where to take all the fasta files) and will produce as output a csv file containing the value of the sum of pairs in the [results/benchmark](./result/benchmark/) folder. Obviously, in order to run this script, the 3 tools must be installed on the computer and can be called correctly from the command line. Using the [GA-DPAMSA_DPAMSA.py](./GA-DPAMSA_DPAMSA.py) script it is possible to perform a comparison benchmark between GA-DPAMSA and DPAMSA (without the use of the genetic algorithm). From the script you can choose to use two different models for the respective calls (for example to compare how one or the other behaves on a differently trained model). The script will produce a csv file with the results of the execution on both models, reporting the value of the sum-of-pairs for both models for each test performed. In the [result/benchmark](./result/benchmark/) folder you can find these csv file. Instead in the folders [/result/reportDPAMSA](./result/reportDPAMSA/) and [/result/reportDPAMSA_GA](./result/reportDPAMSA_GA/) you can find the respective results with the alignment and the value of the sum of pairs. 

## Training
To execute the training of the RL model, you neeed to run the [train.py](./train.py) script. In the first row in the script you can modify the import of the training dataset and you can put any dataset you want, as long as you follow the structure of those present in the [dataset/training_dataset](./datasets/training_dataset/) folder. In the call to the multi_train() function you can change the name of the output model, and the model will be inserted into the [result/weightDPAMSA](./result/weightDPAMSA/) folder at the end of the training.

## Result
Here some graphs will be illustrated regarding the result of some tests performed on DPAMSA (without our Genetic Algorithm), GA-DPAMSA (with our Genetic Algorithm), DPAMSA, ClustalW, ClustalOmega and MSAprobs to demonstrate that our algorithm, on alignment problems of many sequences with more than 30 bases, appears to be better than all of these tools . Every datasets that is used in the evaluation is made up by differets test, and each test corresponds to an alignment problem to be solved.
The settings used for the genetic algorithm for the test are also shown in the following table:

|GA Parameter|Value|
|------------|-----|
|Population|5|
|Number of iteration|3|
|Number of Most fitted individuals to be propagated in the next iteration|2|
|Percentage of individuals to be mutated by iteration|20%|
|Type of crossover used| horizontal crossover|
|Type of mutation used| best fitted individuals|

Obviously the higher the SP score, the better the alignment, so in the case of negative values, the closer we are to 0 the better.


### Results on a 3x30bp dataset
Each test in the dataset will have 3 sequences, where each sequence consists of 30 bases. Note that in this case, both DPAMSA and GA-DPAMSA use the RL model trained on a 3x30bp dataset (see in [training_dataset1_3x30bp](./datasets/training_dataset/training_dataset1_3x30bp.py)), so in this case we only want to test the benefits given to us by the genetic algorithm alone, without testing the working mechanism on a sub-board (the RL agent in this case will work on the whole board). 
In the following figure, you can see the Average sum-of-pairs (SP) score.
![3x30mean](/img/3x30.png)
And below the boxplot based on the SP score.
![3x30box](/img/box-plot-3x30.png)

From the data collected from this first test, we can definitely say that with our genetic algorithm, we have achieved improvements over the DPAMSA tool and we have also better performance than ClustalOmega.

### Results on a 6x30bp dataset
The next results concern the dataset [dataset1\6x30bp](./datasets/training_dataset/training_dataset1_6x30bp.py), where each test has 6 sequences to align and each is made of 30 nucleotides. Here we will first look at the performance of GA-DPAMSA using the RL model trained on 3x30 dataset (so in this case it will use the concept of sub-boards) and then compare it with DPAMSA trained on 6x30 dataset and GA-DPAMSA with 6x30 dataset (so without using sub-boards).
![6x30box](/img/av-6x30.png)

![6x30box](/img/boxplot-6x30.png)

We see how our tool in this case turns out to be the best, with little difference from ClustalW.
In addition, it is important to note that it turns out to be better even than DPAMSA, trained on a 6x30 dataset. This is very important, because it means that we can save time and resources for training the model, and still get very good results (in this case even better).

### Results on a 6x60bp dataset
In this type of test, what we wanted to check is the goodness of the algorithm as the problem size increases, staying with an RL model trained on small problems. In fact, in the results we will now illustrate, the RL model used by GA-DPAMSA is the one trained on 3x30 datasets (which we have seen before to be even slightly better than the one trained on 6x60)
The next two figures show, respectively, the average SP score and the boxplot related to SP.

![6x60box](/img/6x60-av.png)

![6x60box](/img/box-plot-6x60.png)

Analyzing the data in the table and graphs, it becomes clear that again, our tool still comes out on top, with ClustalW very close in terms of alignment quality.

### Results on the various GA-DPAMSA configurations.
In this section, we will illustrate the results obtained with GA-DPAMSA under varying crossover mechanism (horizontal-crossover or vertical-crossover) and mutation (random mutation and best-fitted mutation). The data we will illustrate below are from a test performed with the dataset [dataset2\6x60bp](./datasets/inference_dataset/dataset1_6x60bp.py), with RL model trained on 3x30 dataset. The following images show some comparative graphs of performance at varying settings.

![ga-benc](/img/ga-dpamsa-benc.png)

![ga-benc-box](/img/box-plot-ga-dpamsa.png)

Analyzing the results, we can conclude that as we expected, the most fitted mutation allows us to have an alignment with a higher SP score. This is because at each iteration, we have the RL agent operate it on the individual with the lowest fitness score, in the sub-board with the lowest fitness score, so we always try to improve the various solutions. However, random mutation still turns out to be a good compromise, because it allows us to get the results faster, as we do not need the SP calculation on all the sub-boards to figure out where to operate, and it still offers results that are not very far from the most fitted mutation, in fact there turns out to be a deviation of only about 38 in the average SP score value. As for the different types of crossovers, we did not expect marked differences, and in fact, considering the case of the most fitted mutation, between the two types of crossovers we have a deviation of only 17 points in the SP score. Obviously by repeating the test and changing the parameters of the genetic algorithm the results can vary significantly, in fact with a higher number of interactions it was observed that the random mutation tends to become more distant in terms of SP than the other type.


# How change the RL model?
If you want to use a totally different model of RL, you need to modify the mutation functions in [GA.py](./GA.py).
You have to modify the ```mutation_on_best_fitted_individuals_worst_sub_board(model)``` function and ```random_mutation(model)```. You need to edit only the code below the ```#Perform Mutation on the sub-board with RL``` comment in both functions.

