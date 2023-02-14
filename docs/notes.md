# Notes

* 🧠 Thought
* ❓ Question that can be answered by the data
* 🪨 Decision
* 👎 Disadvantage
* 👍 Advantage
* 🚨 Warning
* 💪 Task
* 💡 Idea
* ⭐️ Improvement (if had more time)

## Reading Brief

* "Categorized"
    * ❓ How many classes?
    * ❓ Are the classes balanced?
    * 🧠 Can an article belong to multiple classes?
        * ❓ i.e. Multi-class vs Multi-label?
* "Short description"
    * 🧠 How short is "short"?
        * ❓ What's the distribution of description length?
* "link to original text is available"
    * 🧠 Going to try to avoid needing the original text.
        * 👎 Complex to scrape, store and process original text.
    * 🧠 Maybe there will be information in the URL too?
        * e.g. `www.huffpost.com/<category>/<slug>.html`
        * 🧠 But do we even want to use this information?
        * 🧠 Assume we'd want to use this model on other websites.
        * 🚨 Would clarify this requirement for a real world assignment.
        * 🪨 Will avoid using this as input to the model to avoid potential over-fit to HuffPost dataset.
    * 🧠 Could use this as the sample unique identifier?
    * 🧠 Can just use the index as the unique identifier.
* "Productionization Plan"
    * 🧠 Apply to a batch of samples (offline) or a stream (online)?
    * 🧠 How will we roll out the model? Canary release and/or A/B testing?
    * 🧠 How will we monitor the model once it's deployed? What metrics to track and when to alert?
    * 🧠 How will we retrain the model? How will we handle data drift? Will we need new classes?
* "Storytelling"
    * 🧠 Visuals might help with this.
    * 🧠 Some kind of model explainability will be useful.
    * 🧠 What are the most important/most differentiating words for each class?
    * 🧠 What is the model "looking at"/"paying attention to"?
* 💪 Read Kaggle page for more context on dataset.

## Reading [Dataset Overview](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

* "210k news headlines" vs "65k"
    * 🧠 Sub-sampled? And if so, how?
* "2012 to 2022"
    * 🧠 Going to be large changes over this time
    * 🧠 How are we going to handle this?
        * 🧠 Assume we'd want to use this model on new news (rather than backfill)
        * 🚨 Would clarify this requirement for a real world assignment.
        * 💡 Split train/test by time, to get better idea of real-world performance.
* "Category"
    * 🧠 Can use as a class label
* "Headline"
    * 🧠 Can use as an input to model
* "Authors"
    * 🧠 Will be correlations of class and author (e.g. more 'politics' from Author #3).
    * 🧠 Will be a confounding variable (via style of writing) in Headline and Short description.
    * 🧠 Very specific to HuffPost. Do we want to use this information?
    * 🧠 Again it depend on usage of model, but assume we'd want to use this model on other websites.
    * 🚨 Would clarify this requirement for a real world assignment.
    * 🪨 Will avoid using this as input to the model to avoid potential over-fit to HuffPost dataset.
* "Link"
    * We've discussed this already.
* "Short description"
    * 🧠 Will likely be one of the most useful inputs to the model.
    * 🧠 Will need to be careful about "unhelpful" correlations in the text here.
        * 🚨 e.g. specific formatting styles between classes (and specific for HuffPost)
* "Date"
    * 🧠 Could be correlations of class and date (e.g. more 'politics' in election years).
    * 👎 We don't want the model to bypass text inputs and just use date.
    * 👍 Can be useful to split train/test by date, given assumption mentioned above.
    * 🪨 Will avoid using this as an input feature to the model.
* "42 news categories"
    * 🧠 A manageable number of classes.
    * 🧠 Observe a class imbalance.
        * x10 samples between #1 class and #15 class.
    * 🧠 Will need to handle this in metrics (and possibly training too).
* 💪 Continue with some Exploratory Data Analysis of provided data.

## Exploratory Data Analysis

* 💪 Zipped/renamed data to prevent it being searchable (after being made public).

```bash
tar -C ./data -czvf ./data/compressed/train.json.tar.gz train.json && rm ./data/train.json
tar -C ./data -czvf ./data/compressed/test.json.tar.gz test.json && rm ./data/test.json
```

* 💪 Added data to Git repo for simplicity (at `data/compressed`)
    * ⭐️ Would add to external storage in practice: via DVC and Git LFS.
* See [eda.ipynb](../notebooks/eda.ipynb) for a pre-rendered notebook.
* Source of notebook is [eda.py](../notebooks/eda.py) (via [Jupytext](https://github.com/mwouts/jupytext)).

## Model Evaluation

* 🧠 We have an imbalanced multi-class classification problem.
* 💡 Use binary classification metrics and apply 'one-vs-all' (OvA). Will then get a metric for each class.
* 🧠 We need a metric that works well with imbalanced data.
    * 👎 Accuracy will be misleading.
    * 👍 Balanced accuracy is a much better metric.
    * 👍 Also F1 score handles imbalanced data well (and combines precision and recall into a single metric).
* 🪨 Will use F1 score as our primary metric (per class). Others can still be computed and reviewed.
* 🧠 Will also need to deal with averaging our per class OvA F1 scores.
* 🧠 Can macro or micro average. Which is best?
* 🧠 Going to assume we'd like good performance across all classes, rather than just the most common classes.
* 🪨 Will use Macro Averaged F1 score as our primary metric. Others can still be computed and reviewed.
* 💡 Can use `imbalanced-learn` for reports.

## Models

* 💪 Create a baseline model
* 💡 Trivial Model A: predict class at random (based on uniform distribution)
* 💡 Trivial Model B: predict most common class
* 💡 Trivial Model C: predict class at random (based on class distribution)
* 🧠 All these trivial models above will give a good ways to test the metrics.
* 💪 Create more realistic models
* 🧠 Should experiment with a traditional, simpler and more interpretable model first.
* 💡 Model D: TF-IDF features with Multinomial Logistic Regression
    * 🧠 Will use `scikit-learn`.
* 🧠 Can then try more complex models.
* 💡 Model E: Use pre-trained word embeddings -> CNN -> Linear -> Softmax Loss
* 🪨 In the interest of time, won't try Model E.
* 💡 Model F: Use pre-trained LLM -> Linear -> Softmax Loss
    * 🧠 Will use `transformers`, `pytorch` and `pytorch-lightning`.
* 💡 Model G: Use OpenAI APIs and we're done (just kidding! 🤑)

## Model Evaluation & Experimentation

🎖 Best at the time
🏆 Best overall

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [RandomClass](../src/text_classification/models/random_label.py) | 0.084 | 0.084 | |
| [MostCommonClass](../src/text_classification/models/most_common_label.py) | 0.047 | 0.047 | |
| [LabelDistribution](../src/text_classification/models/label_distribution.py) | 0.100 | 0.096 | 🎖 |

* 🧠 So trivial baselines can give us a Macro F1 score of ~10%.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.809 | 0.662 | 🎖 All defaults |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.814 | 0.658 | +stop_words="english" |

* 👎 Stop word removal didn't help, so revert.
* ⭐️ Would ideally like to cross-validate these metrics to get an idea of the variance.
* 🧠 Also seeing some over-fitting here (f1_train >> f1_valid).
* 💡 Add more regularization to the Logistic Regression model.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.424 | 0.409 | +C=0.1 |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.796 | 0.658 | +C=0.9 |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.716 | 0.615 | +C=0.5 |

* 👎 Adding C regularization didn't help. Will revert to C=1.0.
* 🧠 We don't see as much over-fitting, but the model is worse.
* 🧠 We applied way too much regularization in C=0.1 case.
* ⭐️ Would ideally like to run a grid search to find the best C value (and other hyper-parameters).
* 🧠 Can also reduce the capacity of the model to avoid over-fit.
* 🧠 Current model uses 46,518 features, so let's set `max_features` to 10,000.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.788 | 0.688 | 🎖 +max_features=10000 |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.756 | 0.658 | +max_features=5000 |

* 🧠 Adding `max_features=10000` helped a bit (but could be due to randomness).
* 🧠 Will try `stop_words="english"` again, now that we have fewer features.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [TfidfWithLR](../src/text_classification/models/tfidf_with_lr.py) | 0.788 | 0.672 | +max_features=10000 +stop_words="english" |

* 🧠 We don't lose much performance, but we will gain better interpretability of features, so we'll keep.
* ❓ What is the model learning? Are there any insights about the training dataset?
    * 📄 See [Interpreting `TfidfWithLR` Model](../notebooks/interpreting_tfidf_with_lr_model.ipynb). [Source](../notebooks/interpreting_tfidf_with_lr_model.py)
* ❓ What's the model's performance for each category?
* ❓ What are some example samples that the model is getting wrong?
    * 📄 See [Evaluating `TfidfWithLR` Model](../notebooks/evaluating_tfidf_with_lr_model.ipynb). [Source](../notebooks/evaluating_tfidf_with_lr_model.py)

* 🧠 Will now try a more modern/complex model.
* 🧠 What model to try?
* 🧠 I've previously done an evaluation on different LLMs and `sentence-transformers/all-MiniLM-L6-v2` gave a good balance between performance and complexity/speed.
* 🚨 What are the throughput requirements? Would verify this on a real world assignment.
* 🧠 Will assume there aren't any throughput requirements for now.

* 🧠 Will pre-compute the embeddings so that we can iterate 'head' models faster.
* 💪 Visualize a sample of the embeddings.
    * 📄 See [Visualize Embeddings](../notebooks/visualize_embeddings.ipynb). [Source](../notebooks/visualize_embeddings.py)
* ⭐️ With more time, could experiment with fine-tuning the LLM, rather than keeping it frozen.
* 🧠 But 65k short text samples would unlikely benefit from fine-tuning the LLM component.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [FrozenLmWithLR](../src/text_classification/models/frozen_lm_with_lr.py) | 0.742 | 0.720 | 🎖 |

* 🚨 Got `lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.`
* 🧠 Will increase the `max_iter` from default of `100` to `500`.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [FrozenLmWithLR](../src/text_classification/models/frozen_lm_with_lr.py) | 0.742 | 0.718 | 🎖 |

* 🧠 Although it converged, the model isn't any better than the previous one.
* 🧠 Will try a shallow neural network head instead. Will be last model to experiment with.

* Created a `ClassificationHead` PyTorch module with the following properties:
    * 5 dense layers (output dimensions: 256, 128, 64, 32 and 10 (num_classes)).
    * ReLU activation between each layer.
    * Adam Optimizer (with momentum of 0.9).
    * OneCycleLR learning rate scheduler (with max learning rate of 0.01).
    * Batch size of 32.
    * Cross-entropy loss.
    * 100 epochs.

* 🧠 Should start by over-fitting a small number of batches to validate the model.
* 👍 Yes, model over-fits 5% of the training data.
    * 1e-5 training loss.
    * ~4 validation loss.

* 🧠 Will now train the model on the full training dataset.
* 🧠 Model trains within first two epochs, so will reduce the number of epochs to ~10.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [FrozenLmWithNN](../src/text_classification/models/frozen_lm_with_nn/model.py) | 0.731 | 0.733 | 🎖 |

* 🚨 Metrics in TensorBoard look better, but 'out of training loop' validation metrics look worse.
* ⭐️ With more time, would investigate why the metrics are different.

* 🧠 Will be conservative and select the model with the best 'out of training loop' validation metrics.
* 🪨 Will choose `FrozenLmWithLR` as the model to 'deploy'.

* 💪 Given this model has been selected, will now calculate the metrics on the hold out test dataset.

* 🧠 Get f1_test of `0.707` which is similar to f1_valid of `0.718`
* 🧠 We've over-fit slightly to validation set through model selection and hyper-parameter tuning, although this is expected.

*UPDATE*

* 🧠 Wanted to try fine-tuning the language model.
* 🧠 Given success of TFIDF baseline, there is clearly useful information in individual words. With current approach of using pre-trained embeddings, we are likely to use general topic information rather than specific words.

* 💪 Overnight training with a single set of hyper-parameters. On CPU only machine, so will be slow.

| Model | Train Macro F1 | Validation Macro F1 | Notes |
| --- | --- | --- | --- |
| [FinetunedLM](../src/text_classification/models/finetuned_lm.py) | 0.811 | 0.781 | 🎖 |

* 🧠 Quite a sizable performance increase.
* 🧠 Some mild over-fitting observed, but not too bad.
* 🧠 Selected the final model from epoch 3/5, after which the model over-fit further.

## Deployment

* Should monitor the character length distribution for `headline` and `short_description`.
* Should enforce an schema for the inputs to the model.
* Can export the neural network model using torch-script for easier deployment.
    * Combine sentence transformer and classification head into a single model.
