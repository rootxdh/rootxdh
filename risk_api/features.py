"""
Shared feature definitions for the Ghost Shopper risk model.

The frontend behaviour tracker (ghost-shopper-demo.html) collects these raw
signals in the browser. Both the training script and the API import from here
so the feature order can never drift between train time and serve time.
"""

from __future__ import annotations

from dataclasses import dataclass

# Order matters: the model is trained on a vector in exactly this order.
FEATURE_NAMES = [
    "time_on_page_s",   # seconds spent before confirming the order
    "desc_opened",      # 1 if the product description was expanded
    "size_selected",    # 1 if a size was chosen
    "scroll_depth",     # max scroll depth reached, 0-100 (%)
    "mouse_moves",      # count of mousemove events
    "keystrokes",       # count of keydown events in the form
    "pastes",           # number of paste events into form fields
    "device_reuse",     # prior orders seen from the same device fingerprint
]


@dataclass
class BehaviorSignals:
    """One session's raw behaviour, as sent by the browser tracker."""

    time_on_page_s: float = 0.0
    desc_opened: bool = False
    size_selected: bool = False
    scroll_depth: float = 0.0
    mouse_moves: int = 0
    keystrokes: int = 0
    pastes: int = 0
    device_reuse: int = 0

    def to_vector(self) -> list[float]:
        """Flatten to the model's expected feature order."""
        return [
            float(self.time_on_page_s),
            float(int(self.desc_opened)),
            float(int(self.size_selected)),
            float(self.scroll_depth),
            float(self.mouse_moves),
            float(self.keystrokes),
            float(self.pastes),
            float(self.device_reuse),
        ]
