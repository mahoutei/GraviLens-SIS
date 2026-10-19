# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).

GRAVILENS_CONFIG = {
    # Architecture — unchanged from v8
    'image_size':              64,
    'patch_size':               8,
    'embed_dim':              192,
    'in_channels':              1,
    'num_classes':              3,
    'num_heads':                8,
    'num_hidden_neurons':      64,
    'num_hidden_layers':        3,
    'num_transformer_blocks':   4,
    'window_size':              4,
    'dropout':                0.1,
    # Training
    'n_epochs':               50,
    'batch_size':              64,
    # [FIX 3] Higher peak LR is now safe with a correctly-wired warmup.
    'initial_lr':            3e-4,   # was 1e-4
    # [FIX 2] 10% warmup budget instead of 40%.
    'warmup_epochs':            5,   # was 20
    # [FIX 4] One optimizer step per batch — same cadence as baselines.
    'accumulation_steps':       1,   # was 4
    'weight_decay':           1e-3,
    # Stability
    'use_ema':               True,
    'ema_decay':           0.9999,
    'use_lookahead':         True,
    # [FIX 5] Faster slow-weight sync (k=3 vs k=5).
    'lookahead_k':              3,   # was hardcoded 5
    'lookahead_alpha':        0.5,
    # [FIX 7] Less noise on gradient signal during early training.
    'label_smoothing':        0.05,  # was 0.1
    # [FIX 6] Preserves more gradient signal (1% -> 5% of param norm).
    'adaptive_clip_factor':   0.05,  # was 0.01
    # [FIX 8] Hold augmentation at zero for first 10 epochs.
    'aug_start':               10,   # new param
    # [FIX 9] Tighter stopping given faster convergence.
    'patience':                25,   # was 30
    'save_every':               5,
}


LENSFORMER_CONFIG = {
    # Architecture (from Velôso et al., 2023)
    'image_size':              64,
    'patch_size':              32,
    'embed_dim':              384,
    'in_channels':              1,
    'num_classes':              3,
    'num_heads':               16,
    'num_hidden_neurons':      64,
    'num_hidden_layers':        3,
    'num_transformer_blocks':   1,
    'dropout':                0.2,
    # Training
    'n_epochs':                50,
    'batch_size':              64,
    'lr':                    1e-4,
    'weight_decay':           1e-4,
    'patience':                25,
    'save_every':               5,
}


LENSPINN_SMALL_CONFIG = {
    # Hyperparameter ablation of Lensformer
    'image_size':              64,
    'patch_size':              16,
    'embed_dim':              384,
    'in_channels':              1,
    'num_classes':              3,
    'num_heads':               16,
    'num_hidden_neurons':      64,
    'num_hidden_layers':        3,
    'num_transformer_blocks':   2,
    'dropout':                0.25,
    # Training
    'n_epochs':                50,
    'batch_size':              64,
    'lr':                    5e-5,
    'weight_decay':           1e-4,
    'patience':                25,
    'save_every':               5,
}


VIT_CONFIG = {
    # Architecture
    'image_size':    64,
    'patch_size':    32,
    'num_classes':    3,
    'dim':         1024,
    'depth':          2,
    'heads':         16,
    'mlp_dim':       16,
    'channels':       1,
    'dropout':      0.1,
    # Training
    'n_epochs':      50,
    'batch_size':    64,
    'lr':          5e-4,
    'weight_decay': 1e-4,
    'patience':      25,
    'save_every':     5,
}


CAIT_CONFIG = {
    # Architecture
    'image_size':    64,
    'patch_size':    32,
    'num_classes':    3,
    'dim':         1024,
    'depth':          2,
    'cls_depth':      1,
    'heads':         16,
    'mlp_dim':       16,
    'channels':       1,
    'dropout':      0.1,
    # Training
    'n_epochs':      50,
    'batch_size':    64,
    'lr':          5e-4,
    'weight_decay': 1e-4,
    'patience':      25,
    'save_every':     5,
}


VITSD_CONFIG = {
    # Architecture
    'image_size':    64,
    'patch_size':    32,
    'num_classes':    3,
    'dim':         1024,
    'depth':          2,
    'heads':         16,
    'mlp_dim':       16,
    'channels':       1,
    'dropout':      0.1,
    # Training
    'n_epochs':      50,
    'batch_size':    64,
    'lr':          5e-4,
    'weight_decay': 1e-4,
    'patience':      25,
    'save_every':     5,
}
