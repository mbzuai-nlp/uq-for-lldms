import logging

from lm_polygraph.utils.dataset import Dataset

from .diffusion_training_stats import DiffusionTrainingStatisticExtractionCalculator

log = logging.getLogger("lm_polygraph")


def load_dataset(args):
    # Adapted from lm-polygraph default_TrainingStatisticExtractionCalculator.py: loads train/background train datasets.
    log.info("=" * 100)
    log.info("Loading train dataset...")

    train_dataset = Dataset.load(
        args.dataset,
        args.text_column,
        getattr(args, "label_column", None),
        batch_size=args.batch_size,
        prompt=getattr(args, "prompt", ""),
        description=getattr(args, "description", ""),
        mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
        n_shot=getattr(args, "n_shot", 5),
        few_shot_split=args.few_shot_split,
        few_shot_prompt=getattr(args, "few_shot_prompt", None),
        instruct=getattr(args, "instruct", None),
        split=args.train_split,
        size=args.size,
        load_from_disk=args.load_from_disk,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    background_train_dataset = None
    if getattr(args, "background_train_dataset", None) is not None:
        background_train_dataset = Dataset.load(
            args.background_train_dataset,
            args.background_train_dataset_text_column,
            args.background_train_dataset_label_column,
            batch_size=args.batch_size,
            data_files=getattr(args, "background_train_dataset_data_files", None),
            split=getattr(args, "background_train_split", "train"),
            size=getattr(
                args,
                "background_train_dataset_size",
                getattr(args, "background_size", getattr(args, "bg_size", None)),
            ),
            load_from_disk=args.background_load_from_disk,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )

    if args.subsample_train_dataset != -1:
        train_dataset.subsample(
            args.subsample_train_dataset,
            seed=(
                int(list(args.seed)[0]) if not isinstance(args.seed, int) else args.seed
            ),
        )
    if (
        background_train_dataset is not None
        and args.subsample_background_train_dataset != -1
    ):
        background_train_dataset.subsample(
            args.subsample_background_train_dataset,
            seed=(
                int(list(args.seed)[0]) if not isinstance(args.seed, int) else args.seed
            ),
        )

    log.info("Done loading train data.")
    return train_dataset, background_train_dataset


def load_stat_calculator(config, builder):
    # Adapted from lm-polygraph default_TrainingStatisticExtractionCalculator.py: builds train-stat calculator from config.
    return_embeddings = getattr(builder, "return_embeddings", False)
    return_token_embeddings = getattr(builder, "return_token_embeddings", False)
    return_lookback_ratios = getattr(builder, "return_lookback_ratios", False)
    return_attention_features = getattr(builder, "return_attention_features", False)
    if getattr(config, "target_metric", None):
        try:
            selected_metric = next(
                m
                for m in builder.generation_metrics
                if m.__str__() == config.target_metric
            )
        except StopIteration:
            raise ValueError(
                f"Metric '{config.target_metric}' not found in generation_metrics: {[m.__str__() for m in builder.generation_metrics]}"
            )
    else:
        selected_metric = None

    train_dataset, background_train_dataset = load_dataset(config)
    return DiffusionTrainingStatisticExtractionCalculator(
        train_dataset,
        background_train_dataset,
        output_attentions=config.output_attentions,
        output_hidden_states=config.output_hidden_states,
        output_diffusion_stats=getattr(config, "output_diffusion_stats", True),
        output_token_embeddings=getattr(config, "output_token_embeddings", False),
        hidden_state_layer=getattr(config, "hidden_state_layer", None),
        return_embeddings=return_embeddings,
        return_token_embeddings=return_token_embeddings,
        return_lookback_ratios=return_lookback_ratios,
        return_attention_features=return_attention_features,
        target_metric=selected_metric,
    )


def diffusion_training_stats_builder(config, builder):
    # Adapted from lm-polygraph utils/factory_stat_calculator.py: delegates to load_stat_calculator.
    return load_stat_calculator(config, builder)
