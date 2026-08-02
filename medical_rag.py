
cat > agent_pipeline.py <<\'PY'
"""
agent_pipeline.py

Multi-agent orchestration: Reasoner, Validator (guideline check), Translator (Bangla).
Provides a safe execution path and conservative fallbacks. Does NOT provide clinical advice.
"""
from typing import Optional, List
import logging

from architecture import SmallCausalTransformer, GuardrailLayer, ClinicalGuardrail
from medical_rag import Retriever

class ReasoningAgent:
    def __init__(self):
        # small local model used for drafting only
        self.model = SmallCausalTransformer()

    def draft(self, prompt: str) -> str:
        # naive tokenization: split by spaces and map to small ids
        toks = prompt.split()[:16]
        ids = [[min(100 + hash(w) % 100, 50256) for w in toks]]
        import torch
        input_ids = torch.tensor(ids, dtype=torch.long)
        logits = self.model(input_ids)
        # greedy decode: pick argmax at last position
        last = logits[0, -1].argmax().item()
        return prompt + "\n[Draft note: model output id=" + str(last) + "]"

class ValidationAgent:
    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()

    def validate(self, draft: str, question: str) -> dict:
        evidence = self.retriever.retrieve(question, top_k=3)
        issues = []
        if not evidence:
            issues.append("No retrieved evidence found; cannot validate recommendations.")
        # naive guideline check: look for keywords
        if any(k in draft.lower() for k in ["diagnos", "must", "guarantee"]):
            issues.append("Draft contains strong medical claims that require guideline support.")
        return {"evidence": evidence, "issues": issues}

class TranslationAgent:
    def __init__(self):
        # placeholder translator; real translation requires heavy seq2seq models
        pass

    def to_bangla(self, text: str) -> str:
        return "(BN) " + text

class MultiAgentSystem:
    def __init__(self, medqa_db: Optional[str] = None):
        self.reasoner = ReasoningAgent()
        self.validator = ValidationAgent(retriever=Retriever(medqa_db=medqa_db))
        self.translator = TranslationAgent()
        self.guardrail = GuardrailLayer()

    def answer(self, question: str) -> dict:
        prompt = f"Clinical question: {question}\nPlease reason step-by-step:" 
        draft = self.reasoner.draft(prompt)
        v = self.validator.validate(draft, question)
        if v.get("issues"):
            return {"status":"blocked", "issues": v.get("issues"), "evidence": v.get("evidence")}
        try:
            safe = self.guardrail.enforce(draft, evidence=[e for e in v.get("evidence", [])])
        except ClinicalGuardrail as e:
            return {"status":"guardrail_blocked", "reason": str(e)}
        bn = self.translator.to_bangla(safe)
        return {"status":"ok", "en": safe, "bn": bn, "evidence": v.get("evidence")}

if __name__ == "__main__":
    system = MultiAgentSystem()
    print(system.answer("What are emergency steps for suspected myocardial infarction?"))
PY'

cat > requirements.txt <<\'PY'
# Minimal requirements for development and testing
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
biopython>=1.79
pytest
PY'

cat > tests/test_smoke.py <<\'PY'
def test_import_modules():
    import architecture, medical_rag, agent_pipeline
    assert hasattr(architecture, 'SmallCausalTransformer')
    assert hasattr(medical_rag, 'Retriever')
    assert hasattr(agent_pipeline, 'MultiAgentSystem')
PY'

cat > .github/workflows/deploy.yml <<\'PY'
name: CI Deploy
on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest -q
      - name: Package placeholder model
        run: |
          mkdir -p artifacts
          # create an empty placeholder model file to be uploaded as artifact
          python - <<PYCODE
with open('artifacts/final_custom_bangla_med_model.pt','wb') as f:
    f.write(b'PLACEHOLDER MODEL FILE - do NOT use clinically')
PYCODE
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: final-custom-bangla-med-model
          path: artifacts/final_custom_bangla_med_model.pt
PY'

# git operations: add, commit, push

git add architecture.py medical_rag.py agent_pipeline.py requirements.txt tests .github/workflows/deploy.yml

COMMIT_MSG="Add safe small-scale medical AI scaffolding; no heavy training or external large-model downloads.\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

git commit -m "$COMMIT_MSG" || echo "No changes to commit"

# Push to origin main

git push origin HEAD:main

# show last commit
git log -1 --pretty=oneline
'
