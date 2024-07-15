import os
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import json

from photolooper.firesting import measure_firesting
from photolooper.powersupply import switch_off, switch_on
from photolooper.status import Command, Status
from photolooper.utils import find_com_port, send_to_arduino
from photolooper.fit import fit_data


def obtain_status(working_directory: Union[str, Path] = "."):
    """
    Obtain the status of the photolooper. This file is written by the
    AutoSuite program.

    Args:
        working_directory (Union[str, Path], optional):
            The working directory. Defaults to ".".

    Returns:
        Status: The status of the photolooper.
    """
    with open(os.path.join(working_directory, "firesting_status.csv"), "r") as handle:
        content = handle.read()

    if "DEGASSING" in content:
        return Status.degassing

    if "PREREACTION-BASELINE" in content:
        return Status.prereaction_baseline

    if "POSTREACTION-BASELINE" in content:
        return Status.postreaction_baseline

    if "REACTION" in content:
        return Status.reaction

    return Status.other


def obtain_command(working_directory: Union[str, Path] = "."):
    """
    Obtain the command of the photolooper. This file is written by the
    AutoSuite program.

    Args:
        working_directory (Union[str, Path], optional):
            The working directory. Defaults to ".".

    Returns:
        Command: The command of the photolooper.
    """
    with open(os.path.join(working_directory, "command.csv"), "r") as handle:
        content = handle.read()

    if "FIRESTING-START" in content:
        return Command.firesting_start

    if "FIRESTING-STOP" in content:
        return Command.firesting_stop

    if "MEASURE" in content:
        return Command.measure

    if "LAMP-ON" in content:
        return Command.lamp_on

    if "LAMP-OFF" in content:
        return Command.lamp_off

    if "FIRESTING-END" in content:
        return Command.firesting_end

    if "PAUSE" in content:
        return Command.pause

    return Command.other


def seed_status_and_command_files(working_directory: Union[str, Path] = "."):
    """
    Seed the status and command files in the working directory.
    Also writes an empty values_for_experiment.csv file.

    Args:
        working_directory (Union[str, Path], optional):
            The working directory. Defaults to ".".
    """
    with open(os.path.join(working_directory, "firesting_status.csv"), "w") as handle:
        handle.write("Start")

    with open(os.path.join(working_directory, "command.csv"), "w") as handle:
        handle.write("FIRESTING-STOP")

    # write instruction CSV with some numbers, but set the run to "false"
    config = {
        "run": "false",
        "volume_water": 10,
        "volume_sacrificial_oxidant": 10,
        "volume_ruthenium_solution": 10,
        "volume_buffer_solution_1": 10,
        "volume_buffer_solution_2": 10,
        "degassing_time": 10,
        "measurement_time": 10,
        "pre_reaction_baseline_time": 15,
        "post_reaction_baseline_time": 5,
    }

    write_instruction_csv(config, working_directory)


def write_break_command(working_directory):
    with open(os.path.join(working_directory, "command.csv"), "w") as handle:
        handle.write("PAUSE")


def write_pause_status(working_directory):
    with open(os.path.join(working_directory, "firesting_status.csv"), "w") as handle:
        handle.write("PAUSE")


def read_yaml(yaml_file: Union[str, Path]) -> list:
    with open(yaml_file, "r") as handle:
        return yaml.safe_load(handle)


def write_instruction_csv(config: dict, instruction_dir: Union[str, Path] = "."):
    column_order = [
        "run",
        "volume_water",
        "volume_sacrificial_oxidant",
        "volume_ruthenium_solution",
        "volume_buffer_solution_1",
        "volume_buffer_solution_2",
        "degassing_time",
        "measurement_time",
        "pre_reaction_baseline_time",
        "post_reaction_baseline_time",
    ]
    df = pd.DataFrame([config])
    df = df[column_order]
    df.to_csv(
        os.path.join(instruction_dir, "values_for_experiment.csv"), index=False, sep=","
    )


def degassing_check(df, chemspeed_working_dir, start=5, end=-1, threshold=5):
    # ensure that the o2 level is decaying
    # If the value of t=0 is less than 5 yM/L bigger than t = 150 s
    df_degas = df[df["status"] == "DEGASSING"]
    start_o2 = df_degas["uM_1"].values[start]
    end_o2 = df_degas["uM_1"].values[end]

    print(
        f"O2 at {df_degas['duration'].values[start]}: {df_degas['uM_1'].values[start]:.3f}"
    )
    print(
        f"O2 at {df_degas['duration'].values[end]}: {df_degas['uM_1'].values[end]:.3f}"
    )
    status = start_o2 - end_o2 > threshold
    if status:
        status = "true"
    else:
        status = "false"

    df_status = pd.DataFrame([{"status": status}])
    df_status.to_csv(
        os.path.join(chemspeed_working_dir, "degassing_ok.csv"), index=False, sep=","
    )

    return status


def main(global_config_path, experiment_config_path):
    previous_command = None
    previous_status = None
    has_measured = False
    global_configs = read_yaml(global_config_path)
    global_configs["chemspeed_working_dir"] = os.path.normpath(
        global_configs["chemspeed_working_dir"]
    )
    global_configs["instruction_dir"] = os.path.normpath(
        global_configs["instruction_dir"]
    )
    configs = read_yaml(experiment_config_path)

    firestring_port = find_com_port(global_configs["firesting_port"]["name"])

    if firestring_port is None:
        raise Exception("🚨 Firesting port not found")

    global_configs["firesting_port"]["port"] = firestring_port

    lamp_port = find_com_port(global_configs["lamp_port"]["name"])
    if lamp_port is None:
        raise Exception("🚨 Lamp port not found")

    global_configs["lamp_port"]["port"] = lamp_port

    arduino_port = find_com_port(global_configs["arduino_port"]["name"])
    if arduino_port is None:
        raise Exception("🚨 arduino port not found")

    global_configs["arduino_port"]["port"] = arduino_port

    # if log directory doesn't exist, create it
    if not os.path.exists(global_configs["log_dir"]):
        os.makedirs(global_configs["log_dir"])

    switch_off(global_configs["lamp_port"]["port"])
    seed_status_and_command_files(global_configs["instruction_dir"])
    for config in configs:
        print("🧪 Working on ", config)
        degassing_checked = False
        # if the results file already exists, warn user and add incrementing number to run name
        if os.path.exists(
            os.path.join(global_configs["log_dir"], f"results_{config['name']}.csv")
        ):
            print(
                f"🚧 Results file for {config['name']} already exists. Adding number to run name"
            )
            config["name"] = (
                config["name"]
                + "_"
                + str(len(os.listdir(global_configs["log_dir"])) + 1)
            )

        write_instruction_csv(config, global_configs["instruction_dir"])
        results = []
        switch_off(global_configs["lamp_port"]["port"])
        df = None
        while True and config["run"] == "true":
            command = obtain_command(
                working_directory=global_configs["chemspeed_working_dir"]
            )
            status = obtain_status(
                working_directory=global_configs["chemspeed_working_dir"]
            )

            if command != previous_command:
                if command == Command.firesting_end:
                    # if the autosuite waits and the python code continues running and reading the firesting_end command, it will continue breaking the executions
                    if df is not None:
                        rate = fit_data(
                            df,
                            os.path.join(
                                global_configs["log_dir"],
                                f"fit_{config['name']}.png",
                            ),
                        )

                        out_dict = {
                            "config": config,
                            "rate": rate,
                            "datetime": df["datetime"].to_list(),
                            "uM_1": df["uM_1"].to_list(),
                            "optical_temperature_2": df[
                                "optical_temperature_2"
                            ].to_list(),
                            "status": df["status"].to_list(),
                        }

                        with open(
                            os.path.join(
                                global_configs["log_dir"],
                                f"results_{config['name']}.json",
                            ),
                            "w",
                        ) as handle:
                            json.dump(out_dict, handle)

                    write_break_command(global_configs["instruction_dir"])
                    write_pause_status(global_configs["instruction_dir"])
                    break

                if command == Command.lamp_off:
                    switch_off(global_configs["lamp_port"]["port"])

                if command == Command.lamp_on:
                    switch_on(
                        global_configs["lamp_port"]["port"],
                        global_configs["arduino_port"]["port"],
                        config["voltage"],
                    )

            if status != previous_status:
                if status == Status.degassing:
                    send_to_arduino(global_configs["arduino_port"]["port"], "1")
                else:
                    if previous_status == Status.degassing:
                        send_to_arduino(global_configs["arduino_port"]["port"], "0")

            if command == Command.firesting_end:
                # if the autosuite waits and the python code continues running and reading the firesting_end command, it will continue breaking the executions
                if df is not None:
                    rate = fit_data(
                        df,
                        os.path.join(
                            global_configs["log_dir"],
                            f"fit_{config['name']}.png",
                            plotting=True,
                        ),
                    )

                    out_dict = {
                        "config": config,
                        "rate": rate,
                        "datetime": df["datetime"].to_list(),
                        "uM_1": df["uM_1"].to_list(),
                        "optical_temperature_2": df["optical_temperature_2"].to_list(),
                        "status": df["status"].to_list(),
                    }

                    with open(
                        os.path.join(
                            global_configs["log_dir"],
                            f"results_{config['name']}.json",
                        ),
                        "w",
                    ) as handle:
                        json.dump(out_dict, handle)

                write_break_command(global_configs["instruction_dir"])
                write_pause_status(global_configs["instruction_dir"])
                break

            if command == Command.lamp_off:
                switch_off(global_configs["lamp_port"]["port"])

            if command == Command.lamp_on:
                switch_on(global_configs["lamp_port"]["port"], config["voltage"])

            if command not in set(
                [Command.firesting_stop, Command.firesting_end, Command.pause]
            ):
                firesting_results = measure_firesting(
                    global_configs["firesting_port"]["port"]
                )
                print(
                    f"uO2: {firesting_results['uM_1']} optical temperature: {firesting_results['optical_temperature_2']}"
                )
                has_measured = True

            else:
                firesting_results = {}

            firesting_results["timestamp"] = time.time()
            firesting_results["datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
            firesting_results["status"] = status.value
            firesting_results["command"] = command.value
            results.append(firesting_results)

            df = pd.DataFrame(results)

            df["duration"] = df["timestamp"] - df["timestamp"].iloc[0]

            if len(df) > 1:
                df["switch"] = [False, False] + [
                    df["status"].values[i - 1] != df["status"].values[i]
                    for i in range(1, len(df) - 1)
                ]
                switch_times = df[df["switch"]]["duration"]
            else:
                switch_times = []
            # df['duration'] = df['duration'].astype('timedelta64[min]')
            if command not in set(
                [Command.firesting_stop, Command.firesting_end, Command.pause]
            ):
                fig, ax = plt.subplots(2, 1, figsize=(16, 8))
                ax[0].scatter(df["duration"], df["uM_1"], s=0.01, marker="o", c="k")
                ax[1].scatter(
                    df["duration"],
                    df["optical_temperature_2"],
                    s=0.05,
                    marker="o",
                    c="k",
                )

                for i, switch_time in enumerate(switch_times):
                    ax[0].axvline(switch_time, c="k")
                    ax[1].axvline(switch_time, c="k")
                    switch_idx = df[df["duration"] == switch_time].index[0]

                    if i == 0:
                        # ax[0].axvspan(
                        #     0,
                        #     switch_time,
                        #     alpha=0.2,
                        #     label=df.iloc[switch_idx -2 ]['status'],
                        #     color=f"C{i}",
                        # )
                        # ax[1].axvspan(
                        #     0,
                        #     switch_time,
                        #     alpha=0.2,
                        #     label=df.iloc[switch_idx -2 ]['status'],
                        #                 color=f"C{i}",
                        # )
                        pass
                    else:
                        ax[0].axvspan(
                            switch_times.values[i - 1],
                            switch_time,
                            alpha=0.2,
                            label=df.iloc[switch_idx - 2]["status"],
                            color=f"C{i}",
                        )
                        ax[1].axvspan(
                            switch_times.values[i - 1],
                            switch_time,
                            alpha=0.2,
                            label=df.iloc[switch_idx - 2]["status"],
                            color=f"C{i}",
                        )
                    if i == len(switch_times) - 1:
                        # fill until the end
                        ax[0].axvspan(
                            switch_time,
                            df.iloc[-1]["duration"],
                            alpha=0.2,
                            label=df.iloc[-1]["status"],
                            color=f"C{i+1}",
                        )
                        ax[1].axvspan(
                            switch_time,
                            df.iloc[-1]["duration"],
                            alpha=0.2,
                            label=df.iloc[-1]["status"],
                            color=f"C{i+1}",
                        )

                ax[1].set_xlabel("time / s")
                ax[0].set_ylabel("O2 / uM/L")
                ax[0].set_title(
                    f'Current O2 concentration, {df["uM_1"].values[-1]:.3f} uM/L'
                )
                ax[1].set_ylabel("T / C")
                ax[0].legend(loc="upper left")
                fig.tight_layout()

                fig.autofmt_xdate()
                fig.savefig(
                    os.path.join(
                        global_configs["log_dir"], f"results_{config['name']}.png"
                    ),
                    dpi=400,
                )
                fig.autofmt_xdate()
                plt.close()
            df.to_csv(
                os.path.join(
                    global_configs["log_dir"], f"results_{config['name']}.csv"
                ),
                index=False,
            )

            if status == Status.degassing and has_measured:
                degassing_frame = df[df["status"] == "DEGASSING"]
                degassing_frame = degassing_frame.dropna(subset=["uM_1"])
                start = degassing_frame["duration"].values[0]
                end = degassing_frame["duration"].values[-1]
                duration = end - start
                if duration > 150 and not degassing_checked:
                    degassing_checked = True
                    degassing_check(
                        degassing_frame, global_configs["chemspeed_working_dir"]
                    )

            time.sleep(global_configs["sleep_time"])

            previous_command = command
            previous_status = status
