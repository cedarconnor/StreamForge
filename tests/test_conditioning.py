from streamforge.diffusion.conditioning import PromptCache, ReferenceCache


def test_prompt_cache_recomputes_only_on_change():
    calls = []

    def fake_encode(p):
        calls.append(p)
        return ("emb", p)

    c = PromptCache(encode_fn=fake_encode)
    a = c.get("neon street")
    b = c.get("neon street")   # cached -> same object
    c.get("forest")            # changed -> recompute
    assert a is b
    assert calls == ["neon street", "forest"]


def test_reference_cache_keys_on_identity():
    calls = []

    def fake_embed(img):
        calls.append(id(img))
        return ("emb", id(img))

    c = ReferenceCache(embed_fn=fake_embed)
    a = object()
    c.get(a)
    c.get(a)        # cached
    b = object()
    c.get(b)        # recompute
    assert len(calls) == 2
