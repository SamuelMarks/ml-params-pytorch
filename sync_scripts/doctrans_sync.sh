#!/usr/bin/env bash

# TODO: A cross-platform YAML config to replace this script, and pass along to `python -m doctrans`

python -m doctrans --version

declare -r mod='ml_params_pytorch/ml_params'
declare -r input_file="$mod"'/type_generators.py'
declare -r output_file="$mod"'/trainer.py'

printf 'Setting type annotations of `TensorFlowTrainer` class to match those in "%s"\n' "$input_file"

python -m doctrans sync --class "$mod"'/config.py' \
                        --class-name 'TrainConfig' \
                        --function "$output_file" \
                        --function-name 'TorchTrainer.train' \
                        --argparse-function "$mod"'/cli.py' \
                        --argparse-function-name 'train_parser' \
                        --truth 'function'

python -m doctrans sync --class "$mod"'/config.py' \
                        --class-name 'LoadDataConfig' \
                        --function "$output_file" \
                        --function-name 'TorchTrainer.load_data' \
                        --argparse-function "$mod"'/cli.py' \
                        --argparse-function-name 'load_data_parser' \
                        --truth 'function'

python -m doctrans sync --class "$mod"'/config.py' \
                        --class-name 'LoadModelConfig' \
                        --function "$output_file" \
                        --function-name 'TorchTrainer.load_model' \
                        --argparse-function "$mod"'/cli.py' \
                        --argparse-function-name 'load_model_parser' \
                        --truth 'function'

printf 'Setting type annotations of `load_data_from_tfds_or_ml_prepare` function to match those in "%s"\n' "$input_file"

python -m doctrans sync_properties \
                        --input-file "$input_file" \
                        --input-eval \
                        --output-file "$output_file" \
                        --input-param 'exposed_activations_keys' \
                        --output-param 'TorchTrainer.train.activation' \
                        --input-param 'exposed_datasets_keys' \
                        --output-param 'TorchTrainer.load_data.dataset_name' \
                        --input-param 'exposed_losses_keys' \
                        --input-param 'exposed_optimizer_lr_schedulers' \
                        --output-param 'TorchTrainer.train.lr_scheduler' \
                        --output-param 'TorchTrainer.train.loss' \
                        --input-param 'exposed_optimizers_keys' \
                        --output-param 'TorchTrainer.train.optimizer'

declare -r datasets="$mod"'/datasets.py'
python -m doctrans gen --name-tpl '{name}Config' \
                       --input-mapping 'ml_params_pytorch.ml_params.type_generators.exposed_datasets' \
                       --type 'argparse' \
                       --output-filename "$datasets"
git add "$datasets"

declare -ra generate=('activations' 'losses' 'optimizer_lr_schedulers' 'optimizers')
IFS=','
printf 'Using "%s" as truth to generate CLIs for %s\n' "$input_file" "${generate[*]}"

for name in ${generate[*]}; do
    fname="$mod"'/'"$name"'.py';
    rm "$fname"
    python -m ml_params_pytorch.ml_params.doctrans_cli_gen "$name" | xargs python -m doctrans gen;
    git add "$fname"
done

fd -HIepy -x sh -c 'autoflake --remove-all-unused-imports -i "$0" && isort --atomic "$0" && python -m black "$0"' {} \;

printf '\nFIN\n'
