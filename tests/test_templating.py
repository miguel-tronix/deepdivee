import pytest
from deepdive.agent.templating import render


def test_render_system_prompt():
    prompt = render("system_prompt.jinja2")
    assert "expert medical AI" in prompt
    assert "retrieve_pubmed_context" in prompt


def test_render_analysis_prompt():
    prompt = render("analysis_prompt.jinja2", intervention="aspirin")
    assert "Intervention: aspirin" in prompt
    assert "retrieve_pubmed_context" in prompt


def test_render_context_chunk():
    prompt = render("context_chunk.jinja2", pmid="12345", title="A Study", content="Some abstract text.")
    assert "PMID: 12345" in prompt
    assert "Title: A Study" in prompt
    assert "Abstract: Some abstract text." in prompt
