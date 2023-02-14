"""
Script for debugging a particular asset (i.e. can use breakpoints).
"""

from dagster import materialize, SourceAsset, DagsterInstance

from text_classification.assets.models import finetuned_lm


if __name__ == "__main__":
    materialize(
        assets=[
            finetuned_lm,
            SourceAsset('train_set'),
            SourceAsset('validation_set'),
            SourceAsset('label_transform')
        ],
        instance=DagsterInstance.get(),
    )
