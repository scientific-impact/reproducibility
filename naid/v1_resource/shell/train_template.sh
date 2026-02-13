DATA_PATH="/mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/NAID/NAID_test_extrainfo.csv"
prompt=0




OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/mc_mistral \
    --checkpoint meta-llama/Meta-Llama-3-8B


# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/mc_LLAMA3 \
# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/mc_qwen7b \
#     --checkpoint qwen-7B/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/mc_qwen-1-5b \
#     --checkpoint  qwen-1.5B/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a


# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/mc_qwen0-5b \
#     --checkpoint qwen-0.5B/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe

python v1_resource/v1_finetune.py \
  --checkpoint meta-llama/Meta-Llama-3-8B \
  --data_path v1_resource/NAIDv1/NAID_train_extrainfo.csv \
  --test_data_path v1_resource/NAIDv1/NAID_test_extrainfo.csv \
  --pairwise \
  --pair_max 200000 \
  --total_epochs 5 \
  --runs_dir /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/ScImpactPredict/official_runs/ \
  --batch_size 16 \
  --load_in_8bit 