TRAIN_FRACTION = 0.8
# i.e. 80% of train source data is used for train set,
# the remaining 20% is used for validation set.

ID_FIELD = "id"
CLASS_FIELD = "category"
CLASS_IDX_FIELD = "category_idx"
FEATURE_FIELDS = [
    "headline",
    "short_description"
]

LANGUAGE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CLASS_LABELS = {
    "A": "A: Art",
    "B": "B: Environment",
    "C": "C: Crime",
    "D": "D: Diversity",
    "E": "E: Relationship",
    "F": "F: Fashion",
    "G": "G: US Politics",
    "H": "H: Foreign Affairs",
    "I": "I: Bizarre",
    "J": "J: Parenting"
}
