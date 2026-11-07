import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from app.config import settings
from app.core.errors import AuthenticationError, PromptInjectionDetectedError


class PromptInjectionDefender:
    """Multi-layered prompt injection and jailbreak detection engine."""

    # Explicit jailbreak and system override patterns
    INJECTION_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        (
            re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|rules|prompts)", re.IGNORECASE),
            "INSTRUCTION_OVERRIDE",
            0.95,
        ),
        (
            re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions|directives|prompts)", re.IGNORECASE),
            "INSTRUCTION_DISREGARD",
            0.95,
        ),
        (
            re.compile(r"(system\s+override|override\s+system\s+prompt|new\s+system\s+directive)", re.IGNORECASE),
            "SYSTEM_OVERRIDE",
            0.90,
        ),
        (
            re.compile(r"(you\s+are\s+now|act\s+as)\s+(DAN|unrestricted|jailbroken|an\s+unfiltered\s+AI)", re.IGNORECASE),
            "ROLE_JAILBREAK",
            0.95,
        ),
        (
            re.compile(r"(print|reveal|output|display|dump)\s+(your\s+)?(initial\s+|original\s+|system\s+)?(prompt|instructions|rules)", re.IGNORECASE),
            "SYSTEM_PROMPT_EXTRACTION",
            0.85,
        ),
        (
            re.compile(r"<\/?(system|instruction|developer|admin)>", re.IGNORECASE),
            "DELIMITER_INJECTION",
            0.80,
        ),
        (
            re.compile(r"```(system|instructions|developer)", re.IGNORECASE),
            "CODEBLOCK_DELIMITER_SMUGGLING",
            0.75,
        ),
        (
            re.compile(r"base64\.b64decode|eval\(|exec\(|os\.system", re.IGNORECASE),
            "CODE_EXECUTION_PAYLOAD",
            0.90,
        ),
    ]

    @classmethod
    def inspect(cls, user_text: str) -> Dict[str, Any]:
        """Inspects user input for prompt injection attempts.
        
        Returns inspection metadata including is_safe, score, and detected threats.
        """
        if not settings.PROMPT_INJECTION_PROTECTION_ENABLED or not user_text:
            return {"is_safe": True, "score": 0.0, "threats": []}

        detected_threats = []
        max_score = 0.0

        for pattern, threat_name, threat_score in cls.INJECTION_PATTERNS:
            match = pattern.search(user_text)
            if match:
                detected_threats.append({
                    "threat": threat_name,
                    "matched_substring": match.group(0),
                    "confidence": threat_score,
                })
                if threat_score > max_score:
                    max_score = threat_score

        # Check for excessive delimiter characters (e.g. repeated === or --- or repeated curly braces)
        delimiter_flood = bool(re.search(r"[-=]{10,}|>{10,}|_{10,}", user_text))
        if delimiter_flood:
            detected_threats.append({
                "threat": "DELIMITER_FLOODING",
                "matched_substring": "Repeated delimiter sequence",
                "confidence": 0.70,
            })
            max_score = max(max_score, 0.70)

        is_safe = max_score < 0.80

        return {
            "is_safe": is_safe,
            "score": round(max_score, 3),
            "threats": detected_threats,
        }

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitizes text by stripping harmful control characters and delimiter tags."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters except standard whitespace
        sanitized = "".join(ch for ch in text if ch in "\n\r\t" or ord(ch) >= 32)
        # Neutralize XML-like system tags
        sanitized = re.sub(r"<\/?(system|instruction|developer|admin)[^>]*>", "[TAG_REMOVED]", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    @classmethod
    def validate_or_raise(cls, user_text: str) -> str:
        """Validates user text and raises PromptInjectionDetectedError if an injection is found."""
        inspection = cls.inspect(user_text)
        if not inspection["is_safe"]:
            threat_names = ", ".join(t["threat"] for t in inspection["threats"])
            raise PromptInjectionDetectedError(
                message=f"Prompt rejected: Potential prompt injection or system override detected ({threat_names}).",
                details=inspection,
            )
        return cls.sanitize(user_text)


class SecurityService:
    """Handles password hashing, token creation, and verification."""

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> str:
        """Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
        if salt is None:
            import os
            salt = os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return f"{base64.b64encode(salt).decode('ascii')}${base64.b64encode(pwd_hash).decode('ascii')}"

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain password against stored PBKDF2 hash."""
        try:
            salt_b64, hash_b64 = hashed_password.split("$", 1)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected_hash = base64.b64decode(hash_b64.encode("ascii"))
            actual_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100000)
            return hmac.compare_digest(expected_hash, actual_hash)
        except Exception:
            return False

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta_seconds: Optional[int] = None) -> str:
        """Generates a secure HMAC-SHA256 signed JWT token."""
        now = int(time.time())
        expires = now + (expires_delta_seconds or (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60))
        payload = {**data, "iat": now, "exp": expires}

        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).rstrip(b"=").decode("ascii")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("ascii")

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(settings.SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")

        return f"{header_b64}.{payload_b64}.{sig_b64}"

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verifies HMAC-SHA256 signature and expiration of JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise AuthenticationError("Invalid token format")

            header_b64, payload_b64, sig_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

            # Pad base64
            padding = 4 - (len(sig_b64) % 4)
            if padding != 4:
                sig_b64 += "=" * padding

            expected_sig = hmac.new(settings.SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
            actual_sig = base64.urlsafe_b64decode(sig_b64.encode("ascii"))

            if not hmac.compare_digest(expected_sig, actual_sig):
                raise AuthenticationError("Invalid token signature")

            # Decode payload
            payload_padding = 4 - (len(payload_b64) % 4)
            if payload_padding != 4:
                payload_b64 += "=" * payload_padding

            payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))

            # Check expiration
            if "exp" in payload and int(time.time()) > payload["exp"]:
                raise AuthenticationError("Token has expired")

            return payload
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
