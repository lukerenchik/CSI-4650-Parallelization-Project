import matplotlib.pyplot as plt
import pandas as pd
import json

guess_number_key = "guess_number"

def load_json_file(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def create_dataframe(json_filepath):

    model_json = load_json_file(json_filepath)

    # This dataframe has a dictionary as a value which is bad
    unstructured_dataframe = pd.DataFrame(model_json)

    # Convert the dictionary of "cpu_usage" data into a DataFrame
    cpu_usage_dataframe = pd.json_normalize(unstructured_dataframe["cpu_usage"])

    cpu_usage_dataframe[guess_number_key] = unstructured_dataframe[guess_number_key]

    return cpu_usage_dataframe

def plot(slower_model_json_path, faster_model_json_path):

    filename1 = f"{slower_model_json_path.rstrip('.pth')}_cpu_usage.json"
    filename2 = f"{faster_model_json_path.rstrip('.pth')}_cpu_usage.json"
    slower_model_cpu_usage_dataframe = create_dataframe(filename1)
    faster_model_cpu_usage_dataframe = create_dataframe(filename2)

    time_columns = ["1s", "2s", "5s", "10s"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    datasets = [
        ("Nonparallelized", slower_model_cpu_usage_dataframe, "r"),
        ("Parallelized", faster_model_cpu_usage_dataframe, "b")
    ]

    # Create subplots
    for row, (graph_title, dataset, color) in enumerate(datasets):
        for column_index, column_name in enumerate(time_columns):
            ax = axes[row, column_index]
            ax.plot(dataset[guess_number_key], dataset[column_name], marker="o", color=color, label=f"CPU Usage at {column_name}")
            ax.set_title(f"{graph_title}: CPU Usage at {column_name}")
            ax.set_xlabel("Guess Number")
            ax.set_ylabel("CPU Usage (%)")
            ax.grid(True)

    all_data = pd.concat([slower_model_cpu_usage_dataframe[time_columns], faster_model_cpu_usage_dataframe[time_columns]])
    y_lower_bound = all_data.min().min() - 1
    y_upper_bound = all_data.max().max() + 1

    # Use the same y-axis across all graphs
    for ax in axes.flat:
        ax.set_ylim(y_lower_bound, y_upper_bound)

    plt.tight_layout()
    plt.savefig("graph.jpg")
    plt.show()
