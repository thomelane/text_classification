from dagster import load_assets_from_modules

from . import datasets
from . import transforms
from . import models
from . import embeddings
from . import predictions


dataset_assets = load_assets_from_modules(
    modules=[datasets],
    group_name="datasets",
)


transform_assets = load_assets_from_modules(
    modules=[transforms],
    group_name="transforms",
)


model_assets = load_assets_from_modules(
    modules=[models],
    group_name="models",
)


embedding_assets = load_assets_from_modules(
    modules=[embeddings],
    group_name="embeddings",
)


predictions_assets = load_assets_from_modules(
    modules=[predictions],
    group_name="predictions",
)
