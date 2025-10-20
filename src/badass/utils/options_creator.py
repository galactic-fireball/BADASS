import pathlib
import sys

UTILS_DIR = pathlib.Path(__file__).resolve().parent
BADASS_DIR = UTILS_DIR.parent

sys.path.insert(0, str(BADASS_DIR))

import schema

print(DEFAULT_OPTIONS_SCHEMA)