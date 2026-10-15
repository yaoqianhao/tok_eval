"""Pydantic request / response models for the tok_eval API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    builtin_corpus_files: int
    sessions: int


class TokenizerInfo(BaseModel):
    session_id: str
    kind: str
    is_fast: bool
    vocab_size: int
    files: List[str]
    warnings: List[str] = Field(default_factory=list)
    special_tokens: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    ok: bool
    tokenizer: Optional[TokenizerInfo] = None
    error: Optional[str] = None


class PlaygroundRequest(BaseModel):
    session_id: str
    text: str = ""
    include_special_tokens: bool = False


class TokenPiece(BaseModel):
    index: int
    token: str
    surface: str = ""  # the raw text slice this token covers (human friendly)
    id: int
    start: int
    end: int
    is_continuation: bool


class PlaygroundResponse(BaseModel):
    ok: bool
    tokens: List[TokenPiece] = Field(default_factory=list)
    ids: List[int] = Field(default_factory=list)
    count: int = 0
    char_count: int = 0
    byte_count: int = 0
    word_count: int = 0
    fertility: float = 0.0
    compression_ratio: float = 0.0
    byte_compression: float = 0.0
    decoded: str = ""
    fidelity_pass: bool = False
    error: Optional[str] = None


class EvaluateRequest(BaseModel):
    session_id: str
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    include_custom: bool = True
    custom_only: bool = False
    run_id: Optional[str] = None


class EvaluateResponse(BaseModel):
    ok: bool
    run_id: str = ""
    tokenizer: Optional[TokenizerInfo] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    charts: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class CompareRequest(BaseModel):
    session_a: str
    session_b: str
    categories: Optional[List[str]] = None
    include_custom: bool = True
    run_id: Optional[str] = None


class CompareResponse(BaseModel):
    ok: bool
    run_id: str = ""
    label_a: str = ""
    label_b: str = ""
    leaderboard: List[Dict[str, Any]] = Field(default_factory=list)
    grouped: Dict[str, Any] = Field(default_factory=dict)
    charts: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class CustomCorpusRequest(BaseModel):
    category: str = "custom"
    subcategory: str = "custom"



class CustomCorpusResponse(BaseModel):
    ok: bool
    saved_to: str = ""
    samples: int = 0
    error: Optional[str] = None
