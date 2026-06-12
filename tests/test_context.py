from tokengate.core.context import LayerContext, LayerDecision


def test_layer_context_defaults():
    ctx = LayerContext(request=object())
    assert ctx.response is None
    assert ctx.decisions == []


def test_layer_decision_stores_fields():
    d = LayerDecision(layer="exact_cache", action="miss", detail={"key": "abc123"})
    assert d.layer == "exact_cache"
    assert d.action == "miss"
    assert d.detail == {"key": "abc123"}


def test_layer_context_short_circuit():
    sentinel = object()
    ctx = LayerContext(request=object(), response=sentinel)
    assert ctx.response is sentinel


def test_decisions_list_is_independent():
    ctx1 = LayerContext(request=object())
    ctx2 = LayerContext(request=object())
    ctx1.decisions.append(LayerDecision(layer="x", action="hit"))
    assert ctx2.decisions == []
