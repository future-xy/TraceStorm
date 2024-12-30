import json
from typing import List, Optional, Tuple

import pandas as pd

from tracestorm.logger import init_logger

logger = init_logger(__name__)


def get_datasets(
    datasets_config_file: str,
) -> Tuple[List[Tuple[str, List[str], int, int]], Optional[str]]:
    # Load datasets configuration file
    try:
        with open(datasets_config_file, "r") as f:
            datasets_config = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"Error: Configuration file '{datasets_config_file}' not found"
        )
        return [], None
    except Exception as e:
        logger.error(f"Error reading '{datasets_config_file}': {e}")
        return [], None

    # Strategy to sort the provided datasets
    sort_strategy = datasets_config.pop("sort", "random")

    # List to store info for each dataset
    datasets = []

    for name, config in datasets_config.items():
        file_path = config.get("file_path")
        field = config.get("field")
        try:
            ratio = int(config.get("select_ratio", 1))
        except ValueError:
            logger.error(f"Invalid 'select_ratio' for '{name}'")
            continue

        if not file_path or not field:
            logger.error(
                f"Missing required 'file_path' or 'field' for '{name}'"
            )
            continue

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            logger.error(f"Error: File '{file_path}' not found.")
            continue
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            continue

        if field not in df.columns:
            logger.error(f"Error: Column '{field}' not found in '{file_path}'.")
            continue

        samples = df[field].tolist()

        # Add the dataset information (name, samples, ratio, number of samples)
        datasets.append((name, samples, ratio, len(samples)))
        logger.info(
            f"{name} loaded with {len(samples)} samples, selection ratio = {ratio}"
        )

    return datasets, sort_strategy
