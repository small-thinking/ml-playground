from modeling.basics.benchmark_kv_cache import compute_decode_cost


def test_decode_cost_matches_closed_form_counts():
    cost = compute_decode_cost(batch_size=2, seq_len=4)

    assert cost.uncached_q_tokens == 20
    assert cost.uncached_kv_tokens == 20
    assert cost.cached_q_tokens == 8
    assert cost.cached_kv_tokens == 8
    assert cost.cached_reused_kv_tokens == 12
    assert cost.uncached_attention_score_pairs == 60
    assert cost.cached_attention_score_pairs == 20


def test_decode_cost_reports_cache_hit_rate_and_reductions():
    cost = compute_decode_cost(batch_size=1, seq_len=8)

    assert cost.cache_hit_rate == 28 / 36
    assert cost.kv_projection_reduction == 36 / 8
    assert cost.attention_score_reduction == 204 / 36
