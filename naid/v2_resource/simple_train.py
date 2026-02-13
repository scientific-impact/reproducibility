#!/usr/bin/env python3
"""
Simple training script with configuration at the top.
Just modify the config section and run: python simple_train.py
"""

import os
import sys
from datetime import datetime

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Training settings
TOTAL_EPOCHS = 1
BATCH_SIZE = 1
MAX_PAIRS = 10000
MAX_LENGTH = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
SEED = 42

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = "q_proj,v_proj"
LOAD_IN_8BIT = False

# Data paths
TRAIN_SET = "v2_resource/NAIDv2/NAIDv2-train-with-cites.csv"
# TRAIN_SET = "v1_resource/NAIDv1/NAID_train_extrainfo.csv"
CKPT_PTH = "meta-llama/Meta-Llama-3-8B"

# Pairwise settings (no-grouping mode)
PW_GROUP_BY_CLUSTER_YEAR = False
PW_CURRICULUM = True
PW_BALANCE = False
PW_CAP_PER_PAPER = -1
PW_USE_WEIGHT = False
PW_MIN_DIFF = 0.05
# PW_TARGET_RATIO = "[0.27, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.00]"
PW_TARGET_RATIO = "[0.03,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.27,0.00]"
# PW_TARGET_RATIO = "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"
# Other settings
NEED_PAIRWISE_EVAL = False
SHUFFLE_TRAIN = True

# =============================================================================
# END CONFIGURATION
# =============================================================================

def main():
    # Create run directory
    DATE_STR = datetime.now().strftime("%m%d_%H%M")
    RUN_DIR = f"./runs/{DATE_STR}_simple"
    os.makedirs(RUN_DIR, exist_ok=True)
    
    # Set up sys.argv to simulate command line arguments
    sys.argv = [
        "simple_train.py",
        "--total_epochs", str(TOTAL_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--max_pairs", str(MAX_PAIRS),
        "--data_path", TRAIN_SET,
        "--checkpoint", CKPT_PTH,
        "--max_length", str(MAX_LENGTH),
        "--lora_r", str(LORA_R),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
        "--target_modules", TARGET_MODULES,
        "--runs_dir", RUN_DIR,
        "--learning_rate", str(LEARNING_RATE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--warmup_ratio", str(WARMUP_RATIO),
        "--seed", str(SEED),
        "--load_in_8bit", str(LOAD_IN_8BIT).lower(),
        "--shuffle_train", str(SHUFFLE_TRAIN).lower(),
        "--gt_field", "TNCSI_SP",
        # Pairwise settings - turn off
        "--pw_group_by_cluster_year", str(PW_GROUP_BY_CLUSTER_YEAR).lower(),
        "--pw_curriculum", str(PW_CURRICULUM).lower(),
        "--pw_balance", str(PW_BALANCE).lower(),
        "--pw_cap_per_paper", str(PW_CAP_PER_PAPER),
        "--pw_use_weight", str(PW_USE_WEIGHT).lower(),
        "--pw_min_diff", str(PW_MIN_DIFF),
        "--pw_target_ratio", str(PW_TARGET_RATIO),
    ]
    
    if NEED_PAIRWISE_EVAL:
        sys.argv.append("--need_pairwise_eval")
    
    print("=" * 80)
    print("SIMPLE TRAINING SCRIPT")
    print("=" * 80)
    print(f"Run directory: {RUN_DIR}")
    print(f"Training data: {TRAIN_SET}")
    print(f"Checkpoint: {CKPT_PTH}")
    print(f"Epochs: {TOTAL_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Max pairs: {MAX_PAIRS}, Max length: {MAX_LENGTH}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Grouping: {'Enabled' if PW_GROUP_BY_CLUSTER_YEAR else 'Disabled (no-grouping mode)'}")
    print("=" * 80)
    
    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Import and run the training
    try:
        from v2_finetune import main as train_main
        train_main()
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {RUN_DIR}")
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
