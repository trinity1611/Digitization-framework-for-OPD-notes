"""Gemma 4 26B A4B IT (google/gemma-4-26B-A4B-it): optional mixture-of-experts model.

About 4B parameters are active per token, but all 25.8B are stored and loaded;
memory planning uses the total (see ModelSpec.weights_gb).
"""
from __future__ import annotations

from .gemma4_31b import Gemma4FamilyAdapter


class Gemma4_26BA4BAdapter(Gemma4FamilyAdapter):
    pass
