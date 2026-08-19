import re
from typing import Dict, List, Optional
from langchain_core.documents import Document
from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult
from guardrails import Guard
from guardrails.errors import ValidationError

INTERNAL_CITATION_PATTERN = re.compile(r"\[Internal:\s*([^\]]+)\]")
WEB_CITATION_PATTERN = re.compile(r"\[Web:\s*([^\]]+)\]")


@register_validator(name="valid-citations", data_type="string")
class ValidCitations(Validator):
    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        metadata = metadata or {}
        valid_sources=metadata.get("valid_sources",set())

        internal_citations = INTERNAL_CITATION_PATTERN.findall(value)
        web_citations = WEB_CITATION_PATTERN.findall(value)

        citations = internal_citations + web_citations

        invalid_citations=[
            citation.strip()
            for citation in citations
            if citation.strip() not in valid_sources
        ]

        if not invalid_citations:
            return PassResult()

        error_message=(
            "The answer contains citations that were not present in the "
            f"evidence for this call: {invalid_citations}"
        )

        return FailResult(
            error_message=error_message
        )


def build_guarded_generation_guard() -> Guard:
    return Guard().use(
        ValidCitations(on_fail="noop")
    )


if __name__ == "__main__":
    valid_sources = {
        "src/graph/router.py", 
        "src/graph/nodes.py",
        "https://python.langchain.com/docs/get_started/introduction"
    }

    test_cases = [
        (
            "all citations valid (internal)",
            "The router classifies intent [Internal: src/graph/router.py] "
            "and dispatches to a node [Internal: src/graph/nodes.py].",
        ),
        (
            "fabricated citation (internal)",
            "The router uses a caching layer for performance "
            "[Internal: src/graph/cache.py].",  
        ),
        (
            "honest 'insufficient info' answer - should NOT fail",
            "The provided context does not contain enough information to answer this question.",
        ),
        (
            "valid citation (web)",
            "LangChain provides tools for this [Web: https://python.langchain.com/docs/get_started/introduction].",
        ),
        (
            "fabricated citation (web)",
            "This is supported in the new API [Web: https://python.langchain.com/fake_path].",
        )
    ]

    guard = build_guarded_generation_guard()

    for label, text in test_cases:
        print(f"--- {label} ---")
        
        result = guard.validate(text, metadata={"valid_sources": valid_sources})
        print(f"  validation_passed: {result.validation_passed}")
        
        if not result.validation_passed:
            # Extract the failure reason exactly where Guardrails stored it
            errors = [
                summary.failure_reason 
                for summary in result.validation_summaries 
                if summary.validator_status == 'fail'
            ]
            print(f"  errors: {errors}")
        print()