import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from app.rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ClaimResult:
    claim_text: str
    status: ClaimStatus
    confidence_score: float
    supporting_chunk_id: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    evidence_snippet: Optional[str] = None
    rationale: str = ""


@dataclass
class VerificationReport:
    is_verified: bool
    confidence_score: float
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    claims: List[ClaimResult]
    hallucination_detected: bool
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaimVerifier:
    """Claim-level response verification engine.
    
    Evaluates individual propositions against retrieved evidence, classifies
    each claim (SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED), and calculates
    an explainable evidence grounding score.
    """

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "is", "are", "was", "were", "what", "which", "who", "whom", "this", "that",
        "these", "those", "how", "why", "when", "where", "can", "could", "should", "would",
        "do", "does", "did", "have", "has", "had", "as", "all", "any", "some", "be", "been",
        "being", "it", "its", "they", "their", "them", "we", "our", "us", "you", "your"
    }

    def __init__(self, min_support_ratio: float = 0.60):
        self.min_support_ratio = min_support_ratio

    def extract_claims(self, answer_text: str) -> List[str]:
        """Splits an answer into atomic claim statements for individual factual evaluation."""
        if not answer_text or not answer_text.strip():
            return []

        # Remove citation tags like [Doc: ..., Page: ...] before claim extraction
        cleaned_text = re.sub(r"\[Doc:[^\]]+\]", "", answer_text).strip()
        # Remove markdown headers before sentence splitting
        cleaned_text = re.sub(r"(?m)^#{1,6}\s+.*$", "", cleaned_text).strip()
        if not cleaned_text:
            return []

        # Split on sentence boundaries (avoiding '1. ' digits) or bullet linebreaks
        sentences = re.split(r"(?<!\b\d\.)(?<!\b\d\d\.)(?<=[.!?])\s+|\n{2,}|\n(?=[-*•]\s+|\d+[\.\)]\s+)", cleaned_text)
        claims: List[str] = []

        for s in sentences:
            s_clean = s.strip()
            # Strip remaining bracket citations if any
            s_clean = re.sub(r"\[.*?\]", "", s_clean).strip()
            # Strip leading list markers: e.g. "1. ", "- ", "* "
            s_clean = re.sub(r"^(?:[-*•]|\d+[\.\)])\s*", "", s_clean).strip()
            # Skip empty, greeting, colon-terminated lead-ins, or pure citation lines
            if len(s_clean) < 15 or s_clean.endswith(":"):
                continue
            if re.match(r"^(based on|according to|as stated in|in summary|hello|hi|here are|the following)\b", s_clean, re.IGNORECASE):
                # Clean prefix boilerplate
                s_clean = re.sub(r"^(based on [^,]+,\s*|according to [^,]+,\s*|here (?:are|is)[^:]*:\s*|the following[^:]*:\s*)", "", s_clean, flags=re.IGNORECASE)
            if s_clean and len(s_clean) >= 15:
                claims.append(s_clean)

        return claims if claims else [cleaned_text]

    def verify_answer(
        self,
        answer_text: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> VerificationReport:
        """Audits answer against retrieved evidence chunks at claim level."""
        # 1. Handle insufficient evidence answers
        if "INSUFFICIENT_EVIDENCE" in answer_text.upper():
            return VerificationReport(
                is_verified=True,
                confidence_score=1.0,
                total_claims=1,
                supported_claims=1,
                partially_supported_claims=0,
                unsupported_claims=0,
                claims=[
                    ClaimResult(
                        claim_text=answer_text,
                        status=ClaimStatus.SUPPORTED,
                        confidence_score=1.0,
                        rationale="System explicitly and accurately recognized insufficient evidence.",
                    )
                ],
                hallucination_detected=False,
            )

        if not retrieved_chunks:
            return VerificationReport(
                is_verified=False,
                confidence_score=0.0,
                total_claims=1,
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=1,
                claims=[
                    ClaimResult(
                        claim_text=answer_text,
                        status=ClaimStatus.UNSUPPORTED,
                        confidence_score=0.0,
                        rationale="Zero retrieved evidence chunks provided for verification.",
                    )
                ],
                hallucination_detected=True,
                rejection_reason="No retrieved evidence available to ground the generated response.",
            )

        claims = self.extract_claims(answer_text)
        claim_results: List[ClaimResult] = []

        supported_count = 0
        partial_count = 0
        unsupported_count = 0

        for claim in claims:
            eval_res = self._evaluate_single_claim(claim, retrieved_chunks)
            claim_results.append(eval_res)

            if eval_res.status == ClaimStatus.SUPPORTED:
                supported_count += 1
            elif eval_res.status == ClaimStatus.PARTIALLY_SUPPORTED:
                partial_count += 1
            else:
                unsupported_count += 1

        total = len(claims)
        # Explainable grounding indicator: (supported + 0.5 * partial) / total
        support_ratio = (supported_count + 0.5 * partial_count) / total if total > 0 else 0.0
        support_ratio = round(min(1.0, max(0.0, support_ratio)), 3)

        hallucination_detected = (unsupported_count > 0) or (support_ratio < self.min_support_ratio)
        is_verified = not hallucination_detected and support_ratio >= self.min_support_ratio

        rejection_reason = None
        if hallucination_detected:
            rejection_reason = (
                f"Response contains {unsupported_count} unsupported claim(s). "
                f"Evidence grounding ratio ({support_ratio}) fell below the acceptable threshold ({self.min_support_ratio})."
            )

        return VerificationReport(
            is_verified=is_verified,
            confidence_score=support_ratio,
            total_claims=total,
            supported_claims=supported_count,
            partially_supported_claims=partial_count,
            unsupported_claims=unsupported_count,
            claims=claim_results,
            hallucination_detected=hallucination_detected,
            rejection_reason=rejection_reason,
            metadata={
                "evidence_chunks_evaluated": len(retrieved_chunks),
                "threshold": self.min_support_ratio,
            },
        )

    def _split_into_clauses(self, claim: str) -> List[str]:
        """Splits a compound claim into constituent factual clauses if applicable."""
        # 1. Numbered or lettered sub-clauses: "1) ... 2) ..." or "1. ... 2. ..."
        if re.search(r"(?:\d+[\.\)]\s+|[a-e][\.\)]\s+)", claim):
            parts = re.split(r"(?:\d+[\.\)]\s+|[a-e][\.\)]\s+)", claim)
            clauses = [p.strip().rstrip(";,") for p in parts if len(p.strip()) >= 10]
            if len(clauses) >= 2:
                return clauses

        # 2. Semicolon-delimited clauses
        if ";" in claim:
            parts = [p.strip().rstrip(",") for p in claim.split(";") if len(p.strip()) >= 10]
            if len(parts) >= 2:
                return parts

        # 3. Coordinate conjunctions separating substantial clauses: ", and ", ", while ", ", whereas "
        clause_pattern = r",\s+(?:and|while|whereas|as well as)\s+"
        if re.search(clause_pattern, claim, re.IGNORECASE):
            parts = re.split(clause_pattern, claim, flags=re.IGNORECASE)
            clauses = [p.strip().rstrip(";,") for p in parts if len(p.strip()) >= 10]
            if len(clauses) >= 2:
                return clauses

        return [claim]

    def _evaluate_single_claim(
        self,
        claim: str,
        chunks: List[RetrievedChunk],
    ) -> ClaimResult:
        """Evaluates a single claim statement against candidate evidence chunks.
        
        Combines single-chunk corroboration with composite evidence grounding
        for multi-chunk synthesis claims while preserving strict numeric and
        factual verification.
        """
        all_words = set(re.findall(r"\b\w{3,}\b", claim.lower()))
        claim_words = all_words - self.STOP_WORDS
        if not claim_words:
            claim_words = all_words

        if not claim_words:
            return ClaimResult(
                claim_text=claim,
                status=ClaimStatus.UNSUPPORTED,
                confidence_score=0.0,
                rationale="Claim contains no verifiable content words.",
            )

        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim))

        # 1. Strict Numeric Presence Check Across All Chunks
        all_numbers_grounded = True
        if claim_numbers:
            grounded_numbers = {
                num for num in claim_numbers
                if any(num in c.content for c in chunks)
            }
            if len(grounded_numbers) < len(claim_numbers):
                all_numbers_grounded = False

        # 2. Single-Chunk Evaluation (preserved baseline)
        best_single_score = 0.0
        best_single_chunk: Optional[RetrievedChunk] = None
        best_single_snippet = None

        for chunk in chunks:
            chunk_content = chunk.content.lower()
            matched_words = [w for w in claim_words if w in chunk_content]
            overlap_ratio = len(matched_words) / len(claim_words)

            if claim_numbers:
                chunk_numbers_match = all(num in chunk.content for num in claim_numbers) if all_numbers_grounded else False
                if chunk_numbers_match:
                    score = overlap_ratio * 0.75 + 0.25
                elif all_numbers_grounded and any(num in chunk.content for num in claim_numbers):
                    score = overlap_ratio * 0.65 + 0.15
                else:
                    score = overlap_ratio * 0.30
            else:
                score = overlap_ratio

            if score > best_single_score:
                best_single_score = score
                best_single_chunk = chunk
                best_single_snippet = chunk.content[:200]

        # 3. Composite Evidence Evaluation
        composite_score = 0.0
        contributing_chunks: List[RetrievedChunk] = []
        chunk_word_matches: Dict[str, Set[str]] = {}

        # Chunks must have substantive overlap to prevent generic word inflation
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            matched = {w for w in claim_words if w in chunk_content}
            overlap = len(matched) / len(claim_words) if claim_words else 0.0
            if (len(matched) >= 2 and overlap >= 0.12) or len(matched) >= 3:
                contributing_chunks.append(chunk)
                chunk_word_matches[chunk.chunk_id] = matched

        if len(contributing_chunks) >= 2:
            aggregate_matched_words: Set[str] = set()
            for words in chunk_word_matches.values():
                aggregate_matched_words.update(words)

            aggregate_overlap = len(aggregate_matched_words) / len(claim_words)

            if claim_numbers:
                if all_numbers_grounded:
                    composite_lexical = aggregate_overlap * 0.75 + 0.25
                else:
                    composite_lexical = aggregate_overlap * 0.50
            else:
                composite_lexical = aggregate_overlap * 0.90

            clauses = self._split_into_clauses(claim)
            if len(clauses) >= 2:
                clause_scores: List[float] = []
                for cl in clauses:
                    cl_all_words = set(re.findall(r"\b\w{3,}\b", cl.lower()))
                    cl_words = cl_all_words - self.STOP_WORDS
                    if not cl_words:
                        cl_words = cl_all_words
                    if not cl_words:
                        continue

                    cl_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", cl))
                    best_cl_score = 0.0
                    for c in chunks:
                        c_lower = c.content.lower()
                        cl_matched = [w for w in cl_words if w in c_lower]
                        cl_ratio = len(cl_matched) / len(cl_words)
                        if cl_numbers:
                            c_num_match = any(num in c.content for num in cl_numbers)
                            s = cl_ratio * 0.75 + 0.25 if c_num_match else cl_ratio * 0.50
                        else:
                            s = cl_ratio
                        if s > best_cl_score:
                            best_cl_score = s
                    clause_scores.append(best_cl_score)

                if clause_scores:
                    min_cl = min(clause_scores)
                    avg_cl = sum(clause_scores) / len(clause_scores)
                    if min_cl < 0.30:
                        composite_clause = min(avg_cl, min_cl * 1.5)
                    else:
                        composite_clause = avg_cl * 0.95
                    composite_score = max(composite_lexical, composite_clause)
                else:
                    composite_score = composite_lexical
            else:
                composite_score = composite_lexical

        # Cap composite score strictly if any claimed number is completely absent from all chunks
        if claim_numbers and not all_numbers_grounded:
            composite_score = min(composite_score, 0.30)

        # 4. Final Score & Supporting Chunk Determination
        if composite_score > best_single_score:
            final_score = composite_score
            best_contributor = max(
                contributing_chunks,
                key=lambda c: len(chunk_word_matches.get(c.chunk_id, set())),
                default=best_single_chunk
            )
            supporting_chunk = best_contributor or best_single_chunk
            supporting_snippet = supporting_chunk.content[:200] if supporting_chunk else None
            is_composite = True
        else:
            final_score = best_single_score
            supporting_chunk = best_single_chunk
            supporting_snippet = best_single_snippet
            is_composite = False

        if claim_numbers and not all_numbers_grounded:
            final_score = min(final_score, 0.30)

        final_score = round(min(1.0, max(0.0, final_score)), 3)

        # 5. Status Classification & Rationale
        if final_score >= 0.65:
            status = ClaimStatus.SUPPORTED
            if is_composite:
                rationale = f"Corroborated across multiple retrieved evidence chunks ({len(contributing_chunks)} chunks)."
            else:
                rationale = "Directly corroborated by retrieved evidence text."
        elif final_score >= 0.35:
            status = ClaimStatus.PARTIALLY_SUPPORTED
            rationale = "Partially corroborated; some entities or details lack direct evidence match."
        else:
            status = ClaimStatus.UNSUPPORTED
            rationale = "Insufficient evidence match in retrieved context (hallucination risk)."

        page_num = None
        if supporting_chunk and hasattr(supporting_chunk, "metadata") and isinstance(supporting_chunk.metadata, dict):
            page_num = supporting_chunk.metadata.get("page_number")

        return ClaimResult(
            claim_text=claim,
            status=status,
            confidence_score=final_score,
            supporting_chunk_id=supporting_chunk.chunk_id if supporting_chunk else None,
            document_title=supporting_chunk.document_title if supporting_chunk else None,
            page_number=page_num,
            evidence_snippet=supporting_snippet,
            rationale=rationale,
        )


# Global verifier singleton
claim_verifier = ClaimVerifier()
