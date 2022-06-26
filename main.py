import yaml

import wandb

from src.train import train


def main():
    wandb.login(key=open('secrets/wandb_key.txt', 'r').read(), relogin=True)

    config = yaml.safe_load(open('config.yaml', 'r'))

    wandb.init(project=config['wandb']['project'],
               name=config['wandb']['name'],
               config=config)

    train(config)

    wandb.finish()


if __name__ == '__main__':
    main()
