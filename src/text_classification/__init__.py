from dagster import Definitions

from .assets import (
    dataset_assets,
    transform_assets,
    model_assets,
    embedding_assets,
    predictions_assets
)


all_assets = [
    *dataset_assets,
    *transform_assets,
    *model_assets,
    *embedding_assets,
    *predictions_assets,
]

defs = Definitions(
    assets=all_assets
)
