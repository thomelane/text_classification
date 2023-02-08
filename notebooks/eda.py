# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
import pandas as pd

from text_classification import defs


# %% [markdown]
# Get source assets.

# %%
train_df: pd.DataFrame = pd.DataFrame(defs.load_asset_value("source_train"))  # type: ignore
test_df: pd.DataFrame = pd.DataFrame(defs.load_asset_value("source_test"))  # type: ignore

# %% [markdown]
# ## Samples

# %%
print(f"Samples in train_df: {len(train_df):,}")
print(f"Samples in test_df: {len(test_df):,}")


# %% [markdown]
# 🧠 ~65% of samples in training set.

# %%
def print_row(row: pd.Series):
    print("#" * 50)
    fields = [str(f) for f in row.index]
    for field in fields:
        print(f"# {field}:")
        print(str(row[field]) + "\n")
        
        
def print_sample_rows(df: pd.DataFrame, n: int = 10):
    df.sample(n=n).apply(print_row, axis=1)  # type: ignore


# %%
print_sample_rows(train_df)

# %% [markdown]
# 🚨 Seeing some missing data (for author and short_description).
#
# 💪 Will want to check missing data ratios at some point.
#
# 🧠 Some structure and metadata included in the text fields. e.g.
#
# `(PHOTOS)`, `Lee Daniels: I Wouldn't Be Where I Am If I Embraced Racism`
#
# 🧠 Could be information in `author` field that we'd be ignoring if we skipped
# the feature. e.g.
#
# ```
# Debbe Daley, Contributor
# Principal and owner at Debbe Daley Designs LLC
# ```
#
# 🧠 Even without headline or description, we could guess the article would be
# about design.
#
# 🧠 Concern with `author` field was authors would write multiple articles in
# the same category, and a model would just learn this mapping, rather than
# learning from the article headlines and short_description. We could
# mask/remove the name and leave the rest, but it's only a partial solution and
# more complex.
#
# ⭐️ Check hypothesis that authors post multiple time (and in same category).

# %% [markdown]
# ## Classes

# %% [markdown]
# ❓ How many classes (or categories) are we dealing with?

# %%
classes_in_train = set(train_df["category"].unique())
print(f"Classes in train_df: {len(classes_in_train)}")

# %%
classes_in_test = set(test_df["category"].unique())
print(f"Classes in test_df: {len(classes_in_test)}")

# %% [markdown]
# ❓ Are they the same classes? Or are there new ones (or subset)
# in the test set?

# %%
assert len(classes_in_train.difference(classes_in_test)) == 0
assert len(classes_in_test.difference(classes_in_train)) == 0


# %% [markdown]
# Yes, they are the same classes.

# %% [markdown]
# ❓ What's the count for each class? Is it a 'balanced' problem?

# %%
def plot_class_counts(df: pd.DataFrame) -> None:
    class_counts = df["category"].value_counts(normalize=True)
    class_counts.sort_index().plot(kind="bar")


# %%
plot_class_counts(train_df)


# %% [markdown]
# 🚨 It's imbalanced. As expected from the dataset summary page on
# Kaggle.
#
# ❓ Is there any difference in class distribution at test/inference time?
#
# 🧠 Would like to assume that test set is representative of inference requests,
# but still want to check a few things there.

# %%
def plot_class_counts_comparision(
    df_1: pd.DataFrame,
    label_1: str,
    df_2: pd.DataFrame,
    label_2: str,
) -> None:
    class_counts_1 = df_1["category"].value_counts(normalize=True)
    class_counts_2 = df_2["category"].value_counts(normalize=True)
    class_counts = pd.concat([
        class_counts_1.rename(label_1),
        class_counts_2.rename(label_2)
    ], axis=1)
    class_counts.sort_index().plot(kind="bar")


# %%
plot_class_counts_comparision(train_df, "train", test_df, "test")

# %% [markdown]
# 🧠 Some differences here, but nothing major (e.g. majority class
# at test time being minority class in training set).
#
# 🧠 We can also see that the class labels are just alphabetical. We don't get
# additional information as to what the classes represent. If we had class
# labels like "Politics", "Sport", etc. we could have possibly used this
# information to bootstrap a simple baseline model (e.g. use word embedding of
# the class label). But we can't.

# %% [markdown]
# ## Missing Data

# %% [markdown]
# ❓ What are the missing data proportions for each field?

# %%
train_df.isna().mean()


# %% [markdown]
# 🧠 No missing data?!
#
# 🧠 Seems suspect given we've seen it. Will likely be due to whitespace being
# counted as not na. Will convert.

# %%
def blank_to_na(string: str):
    string = string.strip()
    if len(string) == 0:
        return pd.NA
    else:
        return string


# %%
train_df = train_df.applymap(blank_to_na)
test_df = test_df.applymap(blank_to_na)

# %% [markdown]
# 🧠 Will want to apply this clean up during training and
# inference.

# %%
train_df.isna().mean()

# %% [markdown]
# 🧠 ~20% of time the `author` field is missing. Another reason to
# skip this field for first version.
#
# 🚨 Also ~8% for `short_description` field, which we had noted as one of the
# most important fields.
#
# 👍 Good to see that category is never missing. And headline very rarely.
#
# 💪 Will look at missing overlap between `headline` and `short_description`
#
# ⭐️ Would look at full cross plot of missing values with more time.

# %%
train_df.isna()[["headline", "short_description"]].all(axis=1).sum()

# %% [markdown]
# 🧠 Only 3 samples out of 65,000 have a missing `headline` and
# `short_description`.
#
# 🧠 A negligible amount, but it's still an edge case to handle.
#
# 🧠 Should we introduce an "Unknown" class? e.g. with class index of `-1`?
#
# 💡 Could also apply to case where model isn't confident about top predicted
# class.

# %% [markdown]
# ## Character Length

# %% [markdown]
# ❓ What's the distribution of character length of `headline` and
# `short_description`.
#
# 🧠 Also interested in token length, but we haven't decided on a model (and
# tokenizer) yet.

# %%
train_df["headline_length"] = train_df["headline"].str.len().fillna(0)
train_df["headline_length"].describe()

# %% [markdown]
# 🧠 Will be easier to visualize this.

# %%
train_df["headline_length"].plot(kind="hist", bins=100)

# %% [markdown]
# 🧠 Max of 320 is quite an outlier. It doesn't seem to be a
# trimmed distribution.
#
# ❓ What are the spikes?

# %%
train_df["headline_length"].value_counts().head(10)

# %% [markdown]
# 🧠 It's not a single value.
#
# ❓ What are some samples for these cases?

# %%
print_sample_rows(train_df.query("headline_length == 120"), n=3)

# %% [markdown]
# 🧠 Can't see anything obviously wrong here. Could just be an
# advisory character count in th UI.
#
# ❓ What about `short_description`?

# %%
train_df["short_description_length"] = train_df["short_description"].str.len().fillna(0)
train_df["short_description_length"].describe()

# %%
train_df["short_description_length"].plot(kind="hist", bins=100)

# %% [markdown]
# 🧠 Similar, but even more pronounced spikes.
#
# ❓ What are these spikes?

# %%
train_df["short_description_length"].value_counts().head(10)

# %% [markdown]
# 🧠 0 is known, because it's the missing values we've seen
# earlier.
#
# 🧠 Otherwise quite a few around samples with `short_description_length` of
# >=120.
#
# ❓ What do these samples look like?

# %%
print_sample_rows(train_df.query("short_description_length == 120"), n=3)

# %% [markdown]
# 🚨 It looks like there's been trimming applied here.
#
# ```
# For advertisers, Christmas is a way to get people to buy by doing whatever it takes. One German company, Edeka, goes all
# ```
#
# 🧠 Although slightly strange that it's at a variety of different character
# lengths.
#
# 💡 Could be quite a good data augmentation strategy.
#
# ❓ Should these cases be removed from the dataset.
#
# 🧠 We ultimately want to optimize robust validation metrics, so the question
# is whether we want them removed from that set.
#
# ❓ Are these cases more common for a certain class?

# %%
plot_class_counts_comparision(
    train_df,
    "train",
    train_df.query("short_description_length >= 120 & short_description_length < 130"),
    "train_len_120_to_130"
)

# %% [markdown]
# 🧠 Yes, but still happens for all classes.
#
# ❓ Are these cases in the test set too?

# %%
test_df["headline_length"] = test_df["headline"].str.len().fillna(0)
test_df["headline_length"].plot(kind="hist", bins=100)

# %% [markdown]
# 🧠 Smoother distribution for test set here. Small drop just
# after 100, but not too concerning.

# %%
test_df["short_description_length"] = test_df["short_description"].str.len().fillna(0)
test_df["short_description_length"].plot(kind="hist", bins=100)

# %% [markdown]
# 🧠 Get spikes for `short_description_length` in test set too.
#
# 🧠 Gives more confidence that train and test are samples from the same
# distribution.
#
# 🧠 Going to assume that we'll get these cases at inference time too.
#
# 🪨 Will leave these cases in the dataset (all splits).
#
# 💪 If time permits, and improvements are needed, will try trimming as a data
# augmentation strategy during training.

# %% [markdown]
# ## Over time

# %% [markdown]
# ❓ What's the distribution of samples over time between train and
# test?

# %%
print(train_df.columns)

# %% [markdown]
# 🧠 Where is the field for date? Assume it's been removed from
# original data source.
#
# 🚨 Will assume the samples have been split by time, but can't verify this.
#
# 🚨 In a real world assignment, I would like to have full knowledge/control of
# splitting process. A very common source of error.
