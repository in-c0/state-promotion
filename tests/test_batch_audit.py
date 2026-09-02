import importlib.util
from pathlib import Path

from state_promotion.pals import Example

SPEC = importlib.util.spec_from_file_location("run_lm_pals", Path(__file__).parents[1] / "scripts" / "run_lm_pals.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + (ord(ch) % 67) for ch in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def make_example(*, segment=0, split="train", relation="durable", version=1, target="ZARIN"):
    return Example(
        stream="retention",
        segment=segment,
        context="CTX",
        key="KEY",
        target=target,
        split=split,
        relation=relation,
        version=version,
    )


def test_actual_tokenization_is_invariant_to_harness_only_metadata():
    tok = TinyTokenizer()
    a = make_example(segment=0, split="train", relation="durable", version=1)
    b = make_example(segment=9, split="test", relation="supersedes", version=42)
    aa = MOD.encoded_example_audit(tok, a)
    bb = MOD.encoded_example_audit(tok, b)
    assert aa["model_input_sha256"] == bb["model_input_sha256"]
    assert aa["model_input"] == bb["model_input"]
    assert aa["audit_metadata"] != bb["audit_metadata"]


def test_online_batch_audit_fails_visible_if_nontrain_source_is_present():
    tok = TinyTokenizer()
    audit = MOD.online_batch_audit(
        tok,
        [make_example(split="train"), make_example(split="test")],
        update_applied=True,
        replay_examples=1,
    )
    assert audit["all_source_splits_train"] is False


def test_eval_gold_target_is_not_privileged_in_model_visible_query():
    tok = TinyTokenizer()
    candidates = ["ZARIN", "VEVEK", "KITAL"]
    a = MOD.eval_query_audit(tok, make_example(split="test", target="ZARIN"), candidates)
    b = MOD.eval_query_audit(tok, make_example(split="test", target="VEVEK"), candidates)
    assert a["model_visible_query_sha256"] == b["model_visible_query_sha256"]
    assert a["audit_metadata"]["gold_target"] != b["audit_metadata"]["gold_target"]
    assert a["audit_metadata"]["gold_target_is_model_privileged"] is False
    assert len(a["candidate_encodings"]) == len(candidates)
