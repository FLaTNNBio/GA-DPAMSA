import datasets.training_dataset.synthetic_dataset_4x101bp as dataset1
import utils
from DPAMSA.env import Environment
from DPAMSA.dqn import DQN
import config
import csv
import os
from tqdm import tqdm
import torch

dataset = dataset1


def main():
    config.DEVICE = torch.device(config.DEVICE_NAME)
    multi_train(dataset, truncate_file=True)


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
    print('\n')


def multi_train(dataset=dataset, start=0, end=-1, truncate_file=False, model_path='model_4x101'):

    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(utils.DPAMSA_REPORTS_PATH, f"{tag}.rpt")
    csv_file_name = os.path.join(utils.CSV_PATH, "DPAMSA_training", f"{tag}.csv")

    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Dataset Name", "Number of Sequences", "Alignment Length", "SP Score", "Exact Matches",
                             "Column Score"])

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

        report = (
            f"#: {name}\n"
            f"AL: {alignment_length}\n"
            f"QTY: {num_sequences}\n"
            f"SP: {sp_score}\n"
            f"EM: {exact_matches}\n"
            f"CS: {column_score}\n"
            f"Alignment:\n{env.get_alignment()}\n\n"
        )

        with open(report_file_name, 'a') as report_file:
            report_file.write(report)

        with open(csv_file_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, alignment_length, num_sequences, sp_score, exact_matches, column_score])

    print(f"\nOperazione completata con successo.")
    print(f"Il file di report è stato salvato in: {utils.DPAMSA_REPORTS_PATH}")
    print(f"Il file CSV è stato salvato in: {utils.CSV_PATH}")


def train(index):
    output_parameters()

    assert hasattr(dataset, "dataset_{}".format(index)), "No such data called {}".format("dataset_{}".format(index))
    data = getattr(dataset, "dataset_{}".format(index))

    print("{}: dataset_{}: {}".format(dataset.file_name, index, data))

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


def inference(dataset=dataset, start=0, end=-1, model_path='model_3x30', truncate_file=True):

    output_parameters()

    tag = os.path.splitext(dataset.file_name)[0]
    report_file_name = os.path.join(utils.DPAMSA_REPORTS_PATH, f"{tag}.rpt")
    csv_file_name = os.path.join(utils.INFERENCE_CSV_PATH, "DPAMSA", f"{tag}.csv")

    if truncate_file:
        with open(report_file_name, 'w'):
            pass
        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Dataset Name", "Number of Sequences", "Alignment Length", "SP Score", "Exact Matches",
                             "Column Score"])

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
            f"#: {name}\n"
            f"AL: {metrics['AL']}\n"
            f"QTY: {metrics['QTY']}\n"
            f"SP: {metrics['SP']}\n"
            f"EM: {metrics['EM']}\n"
            f"CS: {metrics['CS']}\n"
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
    print(f"Il file di report è stato salvato in: {utils.DPAMSA_REPORTS_PATH}")
    print(f"Il file CSV è stato salvato in: {utils.CSV_PATH}")


if __name__ == "__main__":

    main()

    #inference(model_path='model_3x30')
