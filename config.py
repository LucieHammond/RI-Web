import os
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(ROOT_DIR, '.env')

load_dotenv(DOTENV_PATH)
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(ROOT_DIR, 'res', 'Data'))
RES_DIR = os.environ.get('INDEX_DIR', os.path.join(ROOT_DIR, 'res'))
QUERIES_DIR = os.environ.get('QUERIES_DIR', os.path.join(ROOT_DIR, 'res', 'Queries'))
