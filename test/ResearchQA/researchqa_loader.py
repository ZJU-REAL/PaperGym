import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RubricItem:
    rubric_item: str
    type: str
    citation_metadata: Optional[str] = None

@dataclass
class ResearchQAItem:
    id: str
    general_domain: str
    subdomain: str
    field: str
    query: str
    date: str
    rubric: List[RubricItem]

def load_researchqa_data(json_path: str) -> List[ResearchQAItem]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for item in data:
        rubric = [RubricItem(**r) for r in item["rubric"]]
        items.append(ResearchQAItem(
            id=item["id"],
            general_domain=item["general_domain"],
            subdomain=item["subdomain"],
            field=item["field"],
            query=item["query"],
            date=item["date"],
            rubric=rubric
        ))
    return items 