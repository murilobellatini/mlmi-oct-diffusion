"""
All relevant paths stored in constant variables
"""

import pathlib as pl

# local paths
root_path = pl.Path(__file__).parents[2]
DATA_PATH = root_path / 'data'
RAW_DATA_PATH = root_path / 'data/raw'
PROCESSED_DATA_PATH = root_path / 'data/raw'
