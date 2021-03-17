import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="specify the checkpoint path to load",
                    type=str)
parser.add_argument("--save_batches", help="save and reload processed batches", action='store_true')

parser.add_argument("--db_dir", help="root directory of the dataset", type=str)

args = parser.parse_args()

assert (args.db_dir), "Please specify the dataset root folder using --db_dir"