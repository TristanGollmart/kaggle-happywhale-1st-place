# 1st Place Solution of Kaggle Happywhale Competition
This is the knshnb's part of the Preferred Dolphin's solution for [Happywhale - Whale and Dolphin Identification](https://www.kaggle.com/competitions/happy-whale-and-dolphin).

## Dataset
Please prepare dataset according to [input/README.md](input/README.md) and place under `input/`.
```
$ ls -F input
fullbody_test_charm.csv   pseudo_labels/          test_backfin.csv*  train_images/
fullbody_test.csv         README.md               test_images/       yolov5_test.csv
fullbody_train_charm.csv  sample_submission.csv*  train2.csv         yolov5_train.csv
fullbody_train.csv        species.npy*            train_backfin.csv
individual_id.npy*        test2.csv               train.csv
```

## Reproducing the winning score
Before the final training round, we repeated 2 rounds of Step 1-2 for pseudo labeling.
By default, `input/pseudo_labels/round2.csv` (the pseudo labels we created) is specified in the config file so that you can skip the first two rounds.
You can train from scratch by setting `None` in `pseudo_label` field in config files.

### Step 1: Training and inference
By `src/train.py`, we
1. train model by whole train data.
2. inference test data and save results under `result/{exp_name}/-1/`.

Several examples of config files are located in `config/`.

Example: Training and inference efficientnet_b6 and efficientnet_b7
```
python -m src.train --config_path config/efficientnet_b6.yaml --exp_name b6
python -m src.train --config_path config/efficientnet_b7.yaml --exp_name b7
```

### Step 2: Postprocess and ensemble
By `src/ensemble.py`, we
1. calculate mean of the predictions by knn and logit for each model.
2. ensemble predictions of the models specified by `--model_dirs`.
3. save prediction as `submission/{out_prefix}-{new_ratio}-{threshold}.csv`.
4. save pseudo label as `submission/pseudo_label_{out_prefix}.csv`.

Predictions generated by charmq's repository are saved in the same format, so you can ensemble them by just specifying paths to model directories.

Example: Ensemble b6 and b7
```
python -m src.ensemble --model_dirs result/b6/-1 result/b7/-1 --out_prefix b6-b7
```

In our post submission, single model (efficientnet_b7) achieved a score that could rank 3rd place in the final leaderboard.
We also confirmed that ensemble of only two models (efficientnet_b6 and efficientnet_b7) could win 1st place.
Ensembling more backbones and charmq's modesl can achieve even better results.

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [1st Place Solution](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/320192) in Kaggle discussion.
- My teammate [charmq's repository](https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq).
