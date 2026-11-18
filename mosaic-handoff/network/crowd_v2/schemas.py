"""Pydantic models for every network message (explicit ``v`` schema version)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HEX64 = r"^[0-9a-f]{64}$"
KID = r"^[0-9a-f]{16}$"
MAX_B64 = 96 * 1024 * 1024


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Envelope(Strict):
    v: Literal[1] = 1
    pub: str = Field(pattern=HEX64)
    body: dict
    sig: str = Field(pattern=r"^[0-9a-f]{128}$")


class DataEnvelope(Envelope):
    data: str = Field(max_length=MAX_B64)


class RegisterBody(Strict):
    type: Literal["register"]
    name: str = Field(default="", max_length=64)


class WorkBody(Strict):
    type: Literal["work"]
    round: int = Field(ge=1)


class CommitBody(Strict):
    type: Literal["commit"]
    round: int = Field(ge=1)
    beacon: str = Field(pattern=HEX64)
    module: int = Field(ge=0, le=64)
    data_seed: int = Field(ge=0)
    payload_hash: str = Field(pattern=HEX64)
    ef_hash: str = Field(pattern=HEX64)
    tokens: int = Field(ge=0)


class UploadBody(Strict):
    type: Literal["upload"]
    round: int = Field(ge=1)
    payload_hash: str = Field(pattern=HEX64)
    claimed: dict[str, float] = Field(default_factory=dict, max_length=8)


class RevealBody(Strict):
    type: Literal["reveal"]
    round: int = Field(ge=1)
    ef_hash: str = Field(pattern=HEX64)


class AuditBody(Strict):
    type: Literal["audit"]
    round: int = Field(ge=1)
    key: str = Field(pattern=KID)
    payload_hash: str = Field(pattern=HEX64)
    cos: float
    ratio: float


class ForwardBody(Strict):
    """Orchestrator-signed wrapper around a client message forwarded to a peer."""
    type: Literal["forward"]
    kind: Literal["register", "work", "commit", "upload", "reveal"]
    via: str = Field(pattern=KID)
    env: dict
    data: str | None = Field(default=None, max_length=MAX_B64)


class EntriesBody(Strict):
    type: Literal["propose", "final"]
    entries: list[dict] = Field(max_length=4096)
