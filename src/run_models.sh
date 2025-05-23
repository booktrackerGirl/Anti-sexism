#!/bin/bash

echo 'Starting with Llama 3'
python src/get_llama3_antisexist.py --dataset './output_data/llama' --output_file 'llama' --output_folder './output_data' --token_num 12 --temperature 0.3 --num_samples 3
sleep 10

echo 'Starting with Qwen2.5-0.5B'
python src/get_qwen_antisexist.py --dataset './data/prep_labeled/total_data' --output_file 'qwen' --output_folder './output_data' --token_num 12 --temperature 0.3 --num_samples 3

sleep 10 * 60


echo 'Starting with Mistral-7B'
python src/get_mistral_antisexist.py --dataset './output_data/mistral' --output_file 'mistral' --output_folder './output_data' --token_num 12 --temperature 0.3 --num_samples 3
sleep 10

sleep 10 * 60

echo 'Starting with Gemma'
python src/get_gemma_antisexist.py --dataset './data/prep_labeled/total_data' --output_file 'gemma' --output_folder './output_data' --token_num 12 --temperature 0.3 --num_samples 3
sleep 10
