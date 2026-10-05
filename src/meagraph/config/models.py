"""Typed configuration models. Analysis-step configs are added with the steps that use them."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from meagraph.probe.spec import DEFAULT_PROBE


class SessionConfig(BaseModel):
    """Which file, stream and probe make up a session, plus metadata the file does not store."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    stream: str = "raw"
    probe: str = DEFAULT_PROBE
    recording_index: NonNegativeInt = 0
    # The stimulated electrode is not recorded in MCS files. Either one electrode id for every
    # STG output, or a mapping such as {"STG 1": "47"}.
    stim_site: str | dict[str, str] | None = None
