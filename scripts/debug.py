"""
Script for debugging a particular asset (i.e. can use breakpoints).
"""

from dagster import materialize, SourceAsset, DagsterInstance

from text_classification.assets.models import sentence_transformer_nn


if __name__ == "__main__":
    materialize(
        assets=[
            sentence_transformer_nn,
            SourceAsset('train_embeddings'),
            SourceAsset('validation_embeddings'),
            SourceAsset('train_set'),
            SourceAsset('validation_set'),
            SourceAsset('label_transform')
        ],
        instance=DagsterInstance.get(),
    )
