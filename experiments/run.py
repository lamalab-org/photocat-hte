import sys
from pathlib import Path

import argparse


# --- ensure local src/ is used (to get the right photolooper) ---
ROOT = Path(__file__).resolve().parents[1]     # go up from experiments/ → project root
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))                  # give priority to local code


from photolooper.main import main

# verify that the imported main came from the local src folder and not from the installed package
main_file = Path(main.__code__.co_filename).resolve()
print("photolooper.main loaded from:", main_file)

if not str(main_file).startswith(str(SRC)):
    raise RuntimeError(
        f"❌ Wrong photolooper imported!\n"
        f"   Expected inside: {SRC}\n"
        f"   But imported from: {main_file}"
    )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config", type=str, default="../config/experiment.yml"
    )
    
    parser.add_argument("--global_config", type=str, default="../config/setup.yml")
    return parser.parse_args()


def run():
    
    args = parse_args()
    exp_path = Path(args.experiment_config).resolve()
    glob_path = Path(args.global_config).resolve()
    print("Global config path    :", glob_path)
    print("Experiment config path :", exp_path)

    main(str(glob_path), str(exp_path))
    
    
    
    """
    args = parse_args()
    print("➡ Global config     :", args.global_config)
    print("➡ Experiment config :", args.experiment_config)
    main(args.global_config, args.experiment_config)
    """


if __name__ == "__main__":
    run()

