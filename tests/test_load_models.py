from neuralnest.load_models import context_length, model, tokenizer


def test_load_models():
    assert tokenizer is not None
    assert model is not None
    assert isinstance(context_length, int)
