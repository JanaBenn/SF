from dataclasses import dataclass

@dataclass(frozen=True)
class Segment:
    gender: str            # "Male" | "Female"
    income_label: str      # "Low" | "Middle" | "High"
    discount_used: bool    # True | False

    def label(self) -> str:
        return f"{self.gender} | {self.income_label} | Disc={self.discount_used}"