import mainGA
import main
import datasets.dataset2_6x30bp as dataset1
import csv
import config
import os

dataset = dataset1

def run_benchmark():
    mainGA.inference(model_path='model_3x30', tag=dataset.file_name, dataset=dataset)
    main.inference(model_path='model_6x30', tag=dataset.file_name, dataset=dataset)

    csv_filename = "{}.csv".format(dataset.file_name)
    csv_filename = os.path.join(config.benchmark_path,csv_filename)
    report_filename_dpmsa = os.path.join(config.report_path_DPAMSA, "{}.rpt".format(dataset.file_name))
    report_filename_ga = os.path.join(config.report_path_DPAMSA_GA, "{}.rpt".format(dataset.file_name))

    data = {}

    # Write data from only the DPAMSA report
    with open(report_filename_dpmsa, 'r') as report_file:
        dataset_name = None
        sp_value = None
        for line in report_file:
            line = line.strip()
            if line.startswith("NO:"):
                if dataset_name is not None and sp_value is not None:
                    data[dataset_name] = {'DPAMSA_SP': sp_value}
                dataset_name = line.split(":")[1].strip()
            elif line.startswith("SP:"):
                sp_value = line.split(": ")[1].strip()

        if dataset_name is not None and sp_value is not None:
            data[dataset_name] = {'DPAMSA_SP': sp_value}

    # Write data from only the DPAMSA_GA report
    with open(report_filename_ga, 'r') as report_file:
        dataset_name = None
        sp_value = None
        for line in report_file:
            line = line.strip()
            if line.startswith("Dataset name:"):
                if dataset_name is not None and sp_value is not None:
                    if dataset_name in data:
                        data[dataset_name]['GA_SP'] = sp_value
                    else:
                        data[dataset_name] = {'GA_SP': sp_value}
                dataset_name = line.split(":")[1].strip()
            elif line.startswith("SP:"):
                sp_value = line.split(": ")[1].strip()

        if dataset_name is not None and sp_value is not None:
            if dataset_name in data:
                data[dataset_name]['GA_SP'] = sp_value
            else:
                data[dataset_name] = {'GA_SP': sp_value}

    # Write data to the CSV file
    with open(csv_filename, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(["File name", "DPAMSA_SP", "GA_SP"])
        for dataset_name, values in data.items():
            dpamsa_sp = values.get('DPAMSA_SP', '')
            ga_sp = values.get('GA_SP', '')
            writer.writerow([dataset_name, dpamsa_sp, ga_sp])

    print("Benchmark completed!")

# Esegui la funzione
run_benchmark()
