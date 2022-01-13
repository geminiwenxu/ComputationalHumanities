from aitextgen import aitextgen
from pytorch_lightning.loggers import TensorBoardLogger
import glob
import yaml
from pkg_resources import resource_filename
from aitextgen.TokenDataset import TokenDataset


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/config/config.yaml')
    folder_path = resource_filename(__name__, config['file_path']['path'])

    # Load model with aitextgen
    hf_model = "minimaxir/hacker-news"
    ai = aitextgen(model=hf_model, verbose=True)
    ai.to_gpu()

    out_dir = "results/"
    all_files = glob.glob(folder_path + "*.txt")
    for file_name in all_files:
        tokenizer_file = "aitextgen.tokenizer.json"
        # set a custom Tokendataset due to short text with block_size=10
        data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=2)
        ai.train(
            data,
            batch_size=100,
            learning_rate=0.1,
            n_gpu=1,
            seed=27,
            num_steps=3,
            generate_every=100,
            output_dir=out_dir,
            # TensorBoardLogger to track the different experiments and keep the model which performs the best
            loggers=[TensorBoardLogger(out_dir)],
            freeze_layers=True,
            line_by_line=False,
            header=False,
        )


if __name__ == '__main__':
    main()
