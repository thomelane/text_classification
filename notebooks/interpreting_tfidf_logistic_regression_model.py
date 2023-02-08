# %% [markdown]
# # Interpreting `TfidfLogisticRegression` Model

# %% [markdown]
# `TfidfLogisticRegression` has two interpretable components:
#
# * `TfidfVectorizer`
# * `LogisticRegression`
#
# 🧠 Can help us understand the dataset better
#
# 🧠 Can help us spot potential issues in the dataset, before we apply more complex models.
#
# 💪 Start with inspecting/interpreting the `TfidfVectorizer`

# %%
from typing import List

from sklearn.preprocessing import LabelEncoder

from text_classification import defs
from text_classification.data import Samples
from text_classification.models import TfidfLogisticRegression


# %% [markdown]
# Get source assets.

# %%
model: TfidfLogisticRegression = defs.load_asset_value("tfidf_logistic_regression_model")  # type: ignore

# %% [markdown]
# ## Interpret `TfidfVectorizer`

# %% [markdown]
# ❓ What are the most common tokens in the dataset?

# %%
vocab_token_to_id = model.tfidf_vectorizer.vocabulary_
vocab_id_to_token = {v: k for k, v in vocab_token_to_id.items()}

# warning: referencing global variables
def most_common_tokens(top_k: int):
    token_idxs = model.tfidf_vectorizer.idf_.argsort()[:top_k]
    idfs = model.tfidf_vectorizer.idf_[token_idxs]
    tokens = [(vocab_id_to_token[id], idf) for id, idf in zip(token_idxs, idfs)]
    for token, idf in tokens:
        print(f"{idf:.2}: {token}")


# %%
most_common_tokens(10)

# %% [markdown]
# 🧠 "Donald Trump" seems to be one of the most common topics in the dataset.
#
# 🧠 Could be that there is a category for US Politics, and that category has a lot of samples in the dataset. Maybe category G?
#
# 🧠 Given the training set, we can calculate its TF-IDF matrix (similar to an embedding matrix), and inspect that.

# %%
train_set: Samples = defs.load_asset_value("train_set")  # type: ignore


# %%
# warning: referencing global variables
def highest_weighted_tokens(category: str, top_k: int = 10):
    samples = [s for s in train_set if s["category"] == category]
    tfidf_matrix = model.embed(samples)
    token_weights = tfidf_matrix.sum(axis=0)
    token_ids = token_weights.argsort()[-top_k:][::-1]
    tokens = [vocab_id_to_token[token_id] for token_id in token_ids]
    for token in tokens:
        print(token)


# %%
label_transform: LabelEncoder = defs.load_asset_value("label_transform")  # type: ignore

# %%
categories: List[str] = label_transform.label_encoder.classes_  # type: ignore
for category in categories:
    print(f"\n### Category: {category}")
    highest_weighted_tokens(category, top_k=10)

# %% [markdown]
# 🧠 Quite distinct topics are observed.
#
# 👍 Seems that TF-IDF is giving sensible features, that can be used by the `LogisticRegression`.
#
# ❓ Is the `LogisticRegression` model using these features as we would expect?

# %% [markdown]
# ## Interpret `LogisticRegression`


# %%
# warning: referencing global variables
def most_influential_tokens(category: str, top_k: int = 10):
    class_idx = label_transform.label_encoder.transform([category])[0]
    token_ids = model.logistic_regression.coef_[class_idx].argsort()[-top_k:][::-1]
    tokens = [vocab_id_to_token[token_id] for token_id in token_ids]
    for token in tokens:
        print(token)


# %%
for category in categories:
    print(f"\n### Category: {category}")
    most_influential_tokens(category, top_k=10)


# %% [markdown]
# 👍 Yes, `LogisticRegression` is using the TF-IDF vectors in a sensible way.
#
# 🧠 We get a very clear view on which tokens contribute to a prediction for each category.
#
# 🧠 So good that we can take an attempt at manual category labelling.

# %%
class_labels = {
    "A": "Art",
    "B": "Environment",
    "C": "Crime",
    "D": "Diversity",
    "E": "Relationship",
    "F": "Fashion",
    "G": "US Politics",
    "H": "Foreign Affairs",
    "I": "Bizarre",
    "J": "Parenting"
}


# %% [markdown]
# 💡 Can give the model some test inputs to get an intuitive feel for the performance.

# %%
def predict(headline: str, short_description: str):
    y_pred = model.predict([{
        "headline": headline,
        "short_description": short_description
    }])[0]
    class_label = class_labels[categories[y_pred]]
    print(class_label)


# %%
predict(
    headline="Renowned Artist Announces New Collection Inspired by Nature",
    short_description="The latest works by this celebrated artist explore the beauty of the natural world."
)

# %%
predict(
    headline="World Shocked as Giant Hamster Takes Over as Ruler of Tiny European Country",
    short_description="Giant hamster becomes surprise ruler of small European country, leaving the world in disbelief and sparking debates on unconventional leadership."
)

# %% [markdown]
# 🧠 Was trying some examples at the border of categories to explore the decision boundary.
#
# 💪 Should look at failure cases. Will do that in a model evaluation report.
