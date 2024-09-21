import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

INPUT_FOLDER = os.path.join(DATA_FOLDER, 'input')
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'output')

def _create_if_needed(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

_create_if_needed(INPUT_FOLDER)
_create_if_needed(OUTPUT_FOLDER)