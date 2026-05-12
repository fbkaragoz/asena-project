import torch
from train.arch import AsenaConfig, AsenaModel


def test_model_forward_shapes():
    cfg = AsenaConfig(
        vocab_size=300, n_layers=2, n_embd=64, n_head=4, n_kv_heads=2,
        mlp_ratio=2.67, rope_theta=10000.0, tie_embeddings=True, init_std=0.02,
        max_seq_len=64,
    )
    model = AsenaModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_model_loss_decreases_one_step():
    cfg = AsenaConfig(vocab_size=300, n_layers=2, n_embd=64, n_head=4, n_kv_heads=2,
                      mlp_ratio=2.67, rope_theta=10000.0, tie_embeddings=True,
                      init_std=0.02, max_seq_len=64)
    model = AsenaModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]   # loss strictly decreases over 10 steps on random data


def test_param_count_matches_estimator():
    from factory.bounds import estimate_param_count
    cfg = AsenaConfig(vocab_size=24000, n_layers=6, n_embd=384, n_head=6, n_kv_heads=2,
                      mlp_ratio=2.67, rope_theta=10000.0, tie_embeddings=True,
                      init_std=0.02, max_seq_len=1024)
    model = AsenaModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    estimated = estimate_param_count(n_layers=6, n_embd=384, n_head=6, n_kv_heads=2,
                                     mlp_ratio=2.67, vocab_size=24000, tied=True)
    # Estimator should be within 10% of actual
    assert abs(actual - estimated) / actual < 0.10
