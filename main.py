import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="VFAD all-in-one")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/vfad/train/test.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=True,
        choices=["train", "eval"],
    )
    
    ### Loaded from torchrun
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    return args

def main(params, args):
    """Main function for VFAD.
    This function handles different tasks such as training and evaluation.
    It parses the given task and calls the corresponding function.
    """
    task = args.task
    if task == "train":
        framework = params['meta']['name']
        if framework == "vfad":
            from src.vfad.train import main as vfad_train
            vfad_train(params, args)
        else:
            raise NotImplementedError(f"Pretraining for framework {framework} is not implemented.")
    elif task == "eval":
        framework = params['meta']['name']
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    main(params, args)