"""End-to-end sanity tests using the bundled examples/ corpus."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.corpus_loader import clean_email_text, load_documents
from src.model import StylometryModel


def _load():
    target = load_documents("examples/target")
    impostor = load_documents("examples/impostor")
    return target, impostor


def test_corpus_loads():
    target, impostor = _load()
    assert len(target) >= 5
    assert len(impostor) >= 3


def test_clean_strips_quotes_and_sig():
    raw = (
        "here is my actual message\n"
        "> quoted line that is not mine\n"
        "On Mon, someone wrote:\n"
        "more quoted junk\n"
        "-- \n"
        "My Signature Block\n"
    )
    cleaned = clean_email_text(raw)
    assert "here is my actual message" in cleaned
    assert "quoted line" not in cleaned
    assert "Signature Block" not in cleaned


def test_supervised_separates_authors():
    target, impostor = _load()
    model = StylometryModel().fit(target, impostor)
    assert model.mode == "supervised"
    genuine = open("examples/test_genuine.txt").read()
    fake = open("examples/test_impostor.txt").read()
    g = model.score(genuine)["likelihood"]
    f = model.score(fake)["likelihood"]
    # The genuine-style email should score clearly higher than the impostor one.
    assert g > f
    assert g > 0.5
    assert f < 0.5


def test_one_class_orders_correctly():
    target, _ = _load()
    model = StylometryModel().fit(target)  # no impostors -> one-class
    assert model.mode == "one_class"
    genuine = open("examples/test_genuine.txt").read()
    fake = open("examples/test_impostor.txt").read()
    assert model.score(genuine)["likelihood"] > model.score(fake)["likelihood"]
