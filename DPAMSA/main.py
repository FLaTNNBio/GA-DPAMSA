import datasets.training_dataset.synthetic_dataset_4x101bp as dataset1
import utils
from DPAMSA.env import Environment
from DPAMSA.dqn import DQN
import config
import csv
import os
from tqdm import tqdm
import torch

DATASET = dataset1
INFERENCE_MODEL = 'model_3x30'


def main():
    config.DEVICE = torch.device(config.DEVICE_NAME)
    multi_train(DATASET, truncate_file=False)


def output_parameters():
    print("---- DPAMSA parameters ---")
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
    print('\n')


def multi_train(dataset=DATASET, start=0, end=-1, truncate_file=False, model_path='model_4x101'):

    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(config.DPAMSA_REPORTS_PATH, f"{tag}.txt")
    csv_file_name = os.path.join(config.CSV_PATH, "DPAMSA_training", f"{tag}.csv")

    if truncate_file:
        # Se truncate_file è True, tronca i file
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Dataset Name", "Number of Sequences", "Alignment Length", "SP Score", "Exact Matches",
                             "Column Score"])
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

    if truncate_file:
        with open(report_file_name, 'w'):
            pass

    datasets_to_process = dataset.datasets[start:end if end != -1 else len(dataset.datasets)]

    for index, name in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):
        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)

        env = Environment(seqs)
        agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
        p = tqdm(range(config.MAX_EPISODE))
        p.set_description(name)

        try:
            agent.load(model_path + '.pth')
        except:
            pass

        for _ in p:
            state = env.reset()
            while True:
                action = agent.select(state)
                reward, next_state, done = env.step(action)
                agent.replay_memory.push((state, next_state, action, reward, done))
                agent.update()
                if done == 0:
                    break
                state = next_state
            agent.update_epsilon()

        state = env.reset()

        while True:
            action = agent.predict(state)
            _, next_state, done = env.step(action)
            state = next_state
            if 0 == done:
                break
        
        env.padding()
        agent.save(model_path)

        alignment_length = len(env.aligned[0])
        sp_score = env.calc_score()
        exact_matches = env.calc_exact_matched()
        column_score = exact_matches / alignment_length
        num_sequences = len(env.aligned)
        # Crea il report testuale

        report = (
            f"File: {name}\n"
            f"Alignment Length (AL): {alignment_length}\n"
            f"Number of Sequences (QTY): {num_sequences}\n"
            f"Sum of Pairs (SP): {sp_score}\n"
            f"Exact Matches (EM): {exact_matches}\n"
            f"Column Score (CS): {column_score}:.3f\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, alignment_length, num_sequences, sp_score, exact_matches, column_score])

    print(f"\nOperazione completata con successo.")
    print(f"Il file di report è stato salvato in: {config.DPAMSA_REPORTS_PATH}")
    print(f"Il file CSV è stato salvato in: {config.CSV_PATH}")


def train(index):
    output_parameters()

    assert hasattr(DATASET, "dataset_{}".format(index)), "No such data called {}".format("dataset_{}".format(index))
    data = getattr(DATASET, "dataset_{}".format(index))

    print("{}: dataset_{}: {}".format(DATASET.file_name, index, data))

    env = Environment(data)
    agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    p = tqdm(range(config.MAX_EPISODE))

    for _ in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.replay_memory.push((state, next_state, action, reward, done))
            agent.update()
            if done == 0:
                break
            state = next_state
        agent.update_epsilon()

    # Predict
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if 0 == done:
            break

    env.padding()
    print("**********dataset: {} **********\n".format(data))
    print("total length : {}".format(len(env.aligned[0])))
    print("sp score     : {}".format(env.calc_score()))
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / len(env.aligned[0])))
    print("alignment: \n{}".format(env.get_alignment()))
    print("********************************\n")


def inference(dataset=DATASET, start=0, end=-1, model_path=INFERENCE_MODEL, truncate_file=True):

    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(config.DPAMSA_REPORTS_PATH, f"{tag}.txt")
    csv_file_name = os.path.join(config.DPAMSA_INF_CSV_PATH, f"{tag}_DPAMSA_results.csv")

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

    for index, name in enumerate(tqdm(datasets_to_process, desc="Processing Datasets"), start):

        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)

        env = Environment(seqs)
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

        metrics = utils.calculate_metrics(env)

        # Crea il report testuale
        report = (
            f"File: {name}\n"
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
            writer.writerow([name, metrics['AL'], metrics['QTY'], metrics['SP'], metrics['EM'], metrics['CS']])

    print(f"\nOperazione completata con successo.")
    print(f"Il file di report è stato salvato in: {config.DPAMSA_REPORTS_PATH}")
    print(f"Il file CSV è stato salvato in: {config.CSV_PATH}\n\n")


def menu():
    """
    Menu interattivo per selezionare tra training e inferenza.
    """
    while True:
        print("\n====== DPAMSA MENU ======")
        output_parameters()
        print(f"Dataset loaded: {DATASET.file_name}\n\n")
        print("1 - Train the model")
        print("2 - Run inference")
        print("3 - Exit")

        choice = input("Select an option (1/2/3): ").strip()

        if choice == "1":
            print("\nStarting training...")
            config.DEVICE = torch.device(config.DEVICE_NAME)
            multi_train(DATASET)

        elif choice == "2":
            print(f"\nModel selected for inference: {INFERENCE_MODEL}")
            confirm = input("Do you want to proceed with inference? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("\nStarting inference...")
                inference(DATASET, model_path=INFERENCE_MODEL)
            else:
                print("\nInference canceled.")

        elif choice == "3":
            print("\nExiting program. Goodbye!")
            break

        else:
            print("\nInvalid choice, please enter 1, 2, or 3.")


if __name__ == "__main__":
    menu()
