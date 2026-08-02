"""
agent_pipeline.py

Multi-agent orchestration for medical QA.
Agents:
 - ReasoningAgent: produces draft medical reasoning using causal transformer core
 - VerificationAgent: verifies claims using Retriever ensemble and simple checks
 - TranslationAgent: translates verified response to Bangla

Pipeline enforces guardrails and requires citations for any clinical recommendation.
"""

from typing import List, Optional
import logging

from architecture import CausalTransformerCore
from medical_rag import RetrieverEnsemble


class ReasoningAgent:
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.core = CausalTransformerCore(model_name=model_name)

    def draft(self, question: str) -> str:
        prompt = (
            "You are a careful clinical reasoning assistant. When answering, think step-by-step and be conservative.\n"
            f"Question: {question}\nAnswer:"
        )
        return self.core.generate(prompt, max_length=256, do_sample=False)[0]


class VerificationAgent:
    def __init__(self, medqa_db_path: Optional[str] = None):
        self.retriever = RetrieverEnsemble(medqa_db_path)

    def verify(self, draft: str, question: str) -> dict:
        """Verify draft against retrieved evidence. Returns a dict with keys: verified_text, evidence_list, issues.

        This implementation uses simple heuristics: it searches for supporting passages and attaches them.
        A production-grade verifier must run fact-checking models and clinical guideline validators.
        """
        evidence = self.retriever.retrieve(question, top_k=5)
        issues = []
        # simple heuristic: if no evidence found and draft contains recommendations, flag an issue
        if not evidence and any(k in draft.lower() for k in ("recommend", "prescribe", "treat", "should")):
            issues.append("No supporting evidence found in retrieval sources.")
        verified_text = draft
        # Append short evidence summary
        if evidence:
            snippets = []
            for src, id_, txt in evidence:
                snippets.append(f"[{src}:{id_}] {txt[:280].replace('\n',' ')}")
            verified_text = draft + "\n\nEvidence:\n" + "\n".join(snippets[:5])
        return {"verified_text": verified_text, "evidence": evidence, "issues": issues}


class TranslationAgent:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-bn"):
        # Using a Marian-style model to translate EN->BN. If not installed, this class will lazy-load.
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception:
            logging.warning("Translation model not available locally. Install transformers and the model weights to enable translation.")
            self.tok = None
            self.model = None

    def translate_to_bangla(self, text: str) -> str:
        if not (self.tok and self.model):
            # fallback: return the original with a note
            return "(Bangla translation unavailable in this environment)\n" + text
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
        out = self.model.generate(**inputs, max_length=600)
        from transformers import AutoTokenizer
        return self.tok.decode(out[0], skip_special_tokens=True)


class MultiAgentPipeline:
    def __init__(self, reasoning_model: str = "sshleifer/tiny-gpt2", medqa_db_path: Optional[str] = None):
        self.reasoner = ReasoningAgent(model_name=reasoning_model)
        self.verifier = VerificationAgent(medqa_db_path=medqa_db_path)
        self.translator = TranslationAgent()

    def answer(self, question: str) -> dict:
        # Step 1: draft reasoning
        draft = self.reasoner.draft(question)

        # Step 2: verify
        verification = self.verifier.verify(draft, question)

        # Step 3: apply guardrail via underlying core (if verification issues exist, block)
        if verification.get("issues"):
            return {"status": "blocked", "issues": verification.get("issues"), "draft": draft}

        # Step 4: final safe text from the reasoning core (we ask the core to reword conservatively)
        # For simplicity, reuse the draft as final_text
        final_text = verification["verified_text"]

        # Step 5: translate to Bangla
        bangla = self.translator.translate_to_bangla(final_text)

        return {
            "status": "ok",
            "final_text_en": final_text,
            "final_text_bn": bangla,
            "evidence": verification.get("evidence", []),
        }


if __name__ == "__main__":
    pipeline = MultiAgentPipeline()
    q = "What are the red flags for acute stroke and what emergency steps should be taken?"
    res = pipeline.answer(q)
    print(res)
