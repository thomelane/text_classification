# %% [markdown]
# # Visualize Embeddings

# %%
import os

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from text_classification import defs
from text_classification.data import Samples
from text_classification.transforms import LabelTransform
import text_classification.config as cfg


# %% [markdown]
# 🧠 Will load pre-computed embeddings.

# %%
train_set: Samples = defs.load_asset_value("train_set")  # type: ignore
train_embeddings: np.ndarray = defs.load_asset_value("train_embeddings")  # type: ignore
label_transform: LabelTransform = defs.load_asset_value("label_transform")  # type: ignore

# %%
train_df = pd.DataFrame(train_set)
train_df["class_label"] = train_df["category"].apply(lambda e: cfg.CLASS_LABELS[e])


# %% [markdown]
# 🧠 We don't want to show 65,000 samples in TensorBoard's embedding viewer.
#
# 🧠 Will sample to 100 samples per class.

# %%
samples_per_class = 100
sampled_train_df = train_df.groupby("category").sample(samples_per_class)
sampled_idxs = sampled_train_df.index
sampled_embeddings = train_embeddings[sampled_idxs]

# %% [markdown]
# 🧠 Choose what fields to show in TensorBoard.

# %%
metadata_header = ["id", "class_label", "headline", "short_description"]
metadata = sampled_train_df[metadata_header].values.tolist()

# %%
data_root = os.environ["DATA_ROOT"]
assert data_root is not None and len(data_root) > 0
output_path = Path(data_root, "embeddings/train")
if output_path.exists():
    shutil.rmtree(output_path)

# %%
writer = SummaryWriter(output_path)
writer.add_embedding(
    sampled_embeddings,
    metadata=metadata,
    metadata_header=metadata_header
)


# %% [markdown]
# Starting the TensorBoard server, and can view the embeddings [here](http://localhost:6006/#projector).

# %%
# #!tensorboard --logdir {output_path}/

# %% [markdown]
# ## Visualization

# %% [markdown]
# ### UMAP
#
# Colouring by class_label.
#
# * 2D
# * Neighbours: 10
# * Iterations: 500
#
# <img src="./umap.png" alt="umap" style="width: 400px;"/>
#
# 🧠 Get some unsupervised seperation, but would like to have given it more iterations.

# %% [markdown]
# ### T-SNE
#
# Colouring by class_label.
#
# * 2D
# * Perplexity: 5
# * Learning Rate: 1
# * Supervision: 25
# * Iterations: 300
#
# <img src="./t-sne.png" alt="t-sne" style="width: 400px;"/>
#
# 🧠 As expected, we get better seperation with supervision.
#
# 🧠 Shows there potential for learning a good head classifier, but we know that from our earlier models too.
#
# ❓ What are the outliers in the clusters?

# %% [markdown]
# ## Outliers

# %%
def print_row(row: pd.Series):
    print("#" * 50)
    fields = [str(f) for f in row.index]
    for field in fields:
        print(f"# {field}:")
        print(str(row[field]) + "\n")


# %%
print_row(train_df.query('id == 8063').iloc[0])

# %% [markdown]
# 🧠 Should this not be in "F: Fashion"?
#
# 🧠 Could be an incorrect label? Or we manually labelled the classes wrong.

# %%
print_row(train_df.query('id == 11520').iloc[0])

# %% [markdown]
# 🧠 Can this be classified into one class? It's also "J: Parenting" and "E: Relationship".

# %%
print_row(train_df.query('id == 51394').iloc[0])

# %% [markdown]
# 🧠 Also "D: Diversity" class.

# %% [markdown]
# ⭐️ Will stop there for now, but would inspect more outliers given more time.
