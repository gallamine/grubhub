# -*- coding: utf-8 -*-

"""Console script for grubhub."""
import sys
import click
import logging
from grubhub import grubhub
import daiquiri

daiquiri.setup(level=logging.INFO)

logger = daiquiri.getLogger(__name__)


def train(region:str) -> str:
    grubhub.train(region=region)


@click.command()
@click.option('-r', '--region', default=1, help='Data region to train on.')
def main(region: str):
    """Console script for grubhub."""
    click.echo(f"Training the model for region {region}!")
    train(region)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
