from lightning.pytorch.tuner import Tuner

from utils import CLI


def main():
    cli = CLI(run=False)
    # Create a tuner for the trainer
    tuner = Tuner(cli.trainer)

    # finds learning rate automatically
    lr_finder = tuner.lr_find(cli.model, datamodule=cli.datamodule, num_training=100)

    # Plot
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_finder.png")


if __name__ == "__main__":
    main()
