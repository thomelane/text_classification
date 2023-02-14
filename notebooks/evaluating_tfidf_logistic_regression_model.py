# %% [markdown]
# # Evaluating `TfidfLogisticRegression` Model

# %%
import json

import numpy as np

from text_classification import defs
from text_classification.data import Samples
from text_classification.models import TfidfWithLR
from text_classification.metrics import report_from_base64
from text_classification.transforms import extract_label, LabelTransform


# %%
# todo: find a way to retrieve dagster asset metadata programatically, instead of via copying!

f1_train = "0.788"
f1_valid = "0.672"

valid_report_base64 = "ICAgICAgICAgICAgICAgICAgICAgICAgICBwcmUgICAgICAgcmVjICAgICAgIHNwZSAgICAgICAgZjEgICAgICAgZ2VvICAgICAgIGliYSAgICAgICBzdXAKCiAgICAgICAgICAgIEE6IEFydCAgICAgICAwLjc5ICAgICAgMC40MCAgICAgIDEuMDAgICAgICAwLjUzICAgICAgMC42MyAgICAgIDAuMzcgICAgICAgNDI0CiAgICBCOiBFbnZpcm9ubWVudCAgICAgICAwLjc0ICAgICAgMC41MCAgICAgIDAuOTkgICAgICAwLjYwICAgICAgMC43MSAgICAgIDAuNDggICAgICAgNDA3CiAgICAgICAgICBDOiBDcmltZSAgICAgICAwLjY1ICAgICAgMC40NCAgICAgIDAuOTkgICAgICAwLjUyICAgICAgMC42NiAgICAgIDAuNDEgICAgICAgNDA1CiAgICAgIEQ6IERpdmVyc2l0eSAgICAgICAwLjc0ICAgICAgMC43MiAgICAgIDAuOTUgICAgICAwLjczICAgICAgMC44MyAgICAgIDAuNjcgICAgICAyMDA5CiAgIEU6IFJlbGF0aW9uc2hpcCAgICAgICAwLjkxICAgICAgMC44MCAgICAgIDAuOTkgICAgICAwLjg1ICAgICAgMC44OSAgICAgIDAuNzggICAgICAgOTk1CiAgICAgICAgRjogRmFzaGlvbiAgICAgICAwLjgzICAgICAgMC44OSAgICAgIDAuOTcgICAgICAwLjg2ICAgICAgMC45MyAgICAgIDAuODYgICAgICAyMDE0CiAgICBHOiBVUyBQb2xpdGljcyAgICAgICAwLjgwICAgICAgMC45MiAgICAgIDAuOTAgICAgICAwLjg1ICAgICAgMC45MSAgICAgIDAuODIgICAgICA0MDIxCkg6IEZvcmVpZ24gQWZmYWlycyAgICAgICAwLjcyICAgICAgMC4zNiAgICAgIDEuMDAgICAgICAwLjQ4ICAgICAgMC42MCAgICAgIDAuMzMgICAgICAgNDE2CiAgICAgICAgSTogQml6YXJyZSAgICAgICAwLjYzICAgICAgMC4zNyAgICAgIDAuOTkgICAgICAwLjQ2ICAgICAgMC42MCAgICAgIDAuMzQgICAgICAgMzc4CiAgICAgIEo6IFBhcmVudGluZyAgICAgICAwLjc4ICAgICAgMC44OSAgICAgIDAuOTYgICAgICAwLjgzICAgICAgMC45MiAgICAgIDAuODUgICAgICAxOTMxCgogICAgICAgYXZnIC8gdG90YWwgICAgICAgMC43OSAgICAgIDAuNzkgICAgICAwLjk1ICAgICAgMC43OCAgICAgIDAuODYgICAgICAwLjc0ICAgICAxMzAwMAo="
valid_report = report_from_base64(valid_report_base64)
print(valid_report)


# %% [markdown]
# * Best Precision: "E: Relationship" 91%
# * Worst Precision: "I: Bizarre" 63%
#
# * Best Recall: "G: US Politics" 92%
# * Worst Recall: "H: Foreign Affairs" 36%
#     
# 🧠 Quite different performance between categories, that's hidden by a (macro) averaged F1 Score.
#
# ❓ Is performance correlated to number of samples per category?
#
# 🧠 Yes, majority class in training set is "G: US Politics". And "I: Bizarre" is a minority classes.
#
# 💡 Could try balancing the classes.
#
# 💪 Add undersampling (as simplest approach) and retest.

# %%
# after undersampling

f1_train = "0.883"
f1_valid = "0.652"

valid_report_base64 = "ICAgICAgICAgICAgICAgICAgICAgICAgICBwcmUgICAgICAgcmVjICAgICAgIHNwZSAgICAgICAgZjEgICAgICAgZ2VvICAgICAgIGliYSAgICAgICBzdXAKCiAgICAgICAgICAgIEE6IEFydCAgICAgICAwLjM5ICAgICAgMC43MCAgICAgIDAuOTYgICAgICAwLjUwICAgICAgMC44MiAgICAgIDAuNjYgICAgICAgNDI0CiAgICBCOiBFbnZpcm9ubWVudCAgICAgICAwLjQ4ICAgICAgMC43NSAgICAgIDAuOTcgICAgICAwLjU4ICAgICAgMC44NSAgICAgIDAuNzEgICAgICAgNDA3CiAgICAgICAgICBDOiBDcmltZSAgICAgICAwLjQyICAgICAgMC43MyAgICAgIDAuOTcgICAgICAwLjU0ICAgICAgMC44NCAgICAgIDAuNjkgICAgICAgNDA1CiAgICAgIEQ6IERpdmVyc2l0eSAgICAgICAwLjc3ICAgICAgMC41OCAgICAgIDAuOTcgICAgICAwLjY2ICAgICAgMC43NSAgICAgIDAuNTQgICAgICAyMDA5CiAgIEU6IFJlbGF0aW9uc2hpcCAgICAgICAwLjg1ICAgICAgMC44MiAgICAgIDAuOTkgICAgICAwLjg0ICAgICAgMC45MCAgICAgIDAuODAgICAgICAgOTk1CiAgICAgICAgRjogRmFzaGlvbiAgICAgICAwLjg4ICAgICAgMC44MSAgICAgIDAuOTggICAgICAwLjg0ICAgICAgMC44OSAgICAgIDAuNzggICAgICAyMDE0CiAgICBHOiBVUyBQb2xpdGljcyAgICAgICAwLjkyICAgICAgMC43MSAgICAgIDAuOTcgICAgICAwLjgwICAgICAgMC44MyAgICAgIDAuNjcgICAgICA0MDIxCkg6IEZvcmVpZ24gQWZmYWlycyAgICAgICAwLjQwICAgICAgMC43NCAgICAgIDAuOTYgICAgICAwLjUyICAgICAgMC44NCAgICAgIDAuNzAgICAgICAgNDE2CiAgICAgICAgSTogQml6YXJyZSAgICAgICAwLjMyICAgICAgMC42MyAgICAgIDAuOTYgICAgICAwLjQyICAgICAgMC43OCAgICAgIDAuNTkgICAgICAgMzc4CiAgICAgIEo6IFBhcmVudGluZyAgICAgICAwLjgxICAgICAgMC44MSAgICAgIDAuOTcgICAgICAwLjgxICAgICAgMC44OSAgICAgIDAuNzcgICAgICAxOTMxCgogICAgICAgYXZnIC8gdG90YWwgICAgICAgMC43OSAgICAgIDAuNzMgICAgICAwLjk3ICAgICAgMC43NSAgICAgIDAuODQgICAgICAwLjY5ICAgICAxMzAwMAo="
valid_report = report_from_base64(valid_report_base64)
print(valid_report)

# %% [markdown]
# 👎 Worse validation metrics after undersampling
#
# 🧠 Minority classes have better recall, but worse precision.
#
# 🧠 Most likely due to incorrect `LogisticRegression` intercepts.
#
# 🧠 Will revert to using the original unbalanced training dataset.
#
# ❓ What are some examples that the model gets wrong?

# %%
model: TfidfWithLR = defs.load_asset_value("tfidf_logistic_regression_model")  # type: ignore
validation_set: Samples = defs.load_asset_value("validation_set")  # type: ignore
label_transform: LabelTransform = defs.load_asset_value("label_transform")  # type: ignore

# %%
X, y = extract_label(validation_set, label_transform)

# %%
y_pred = model.predict(X)

# %%
# get the indices of the samples that were misclassified
misclassified_idxs = np.where(y != y_pred)[0].tolist()

# %%
for i in misclassified_idxs[10:20]:
    sample = X[i].copy()
    sample["label_true"] = label_transform.idx_to_label(y[i])
    sample["label_pred"] = label_transform.idx_to_label(y_pred[i])
    print(json.dumps(sample, indent=4) + '\n')

# %% [markdown]
# ```
# {
#     "headline": "Cupid's Arrow",
#     "short_description": "I remember, as a child, desperately wishing Cupid would hit me with his arrow so I'd fall instantly in love and live happily ever after. Cupid let me down.",
#     "id": 16179,
#     "label_true": "E: Relationship",
#     "label_pred": "J: Parenting"
# }
# ```
#
# 🧠 Can see how a bag-of-words approach could get confused here.
#
# 🧠 Contains the word "child" which was a highly weighted for the "J: Parenting" class.
#
# 🧠 "Cupid" have been filtered out by `max_features=10000`.
#
# ❓ Is "cupid" in the vocabulary?

# %%
"cupid" in model.tfidf_vectorizer.vocabulary_

# %% [markdown]
# 🧠 A difficult tradeoff between adding more features (and over-fitting) and limiting them (and missing cases like this).

# %% [markdown]
# ```
# {
#     "headline": "Texas Adoption Bill Could Allow Anti-Gay, Religious Discrimination",
#     "short_description": "Critics of the bill say it legitimizes discrimination with taxpayer money.",
#     "id": 41233,
#     "label_true": "G: US Politics",
#     "label_pred": "D: Diversity"
# }
# ```
#
# 🚨 We see cases that could be the predicted class, even though ground truth label says otherwise.
#
# 🧠 Is this problem misspecificed? Is it really a multi-label problem (but our dataset only gives a single class).
#
# 🧠 Would the downstream application need a single class? Or would it support multi-label?
#
# 🧠 Where is this model being used? Just for search? In which case multi-label might work.
#
# 🚨 Would clarift these requirements in a real world assignment.
#
# ⭐️ Will focus on multi-class given time constaints, but would be interesting to map to a multi-label problem.
#
# 🧠 Some multi-label methods don't need complete labels to learn.
#
# ⭐️ Would also like to generate a confusion matrix to see which classes are being confused with each other.

# %%
