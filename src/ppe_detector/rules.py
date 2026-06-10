from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    id: int | None = None

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class PersonPPEStatus:
    person: Detection
    helmet_ok: bool
    vest_ok: bool
    violations: List[str]
    matched_helmet: Detection | None = None
    matched_vest: Detection | None = None


@dataclass
class RuleConfig:
    head_zone_ratio: float = 0.45
    torso_y1_ratio: float = 0.25
    torso_y2_ratio: float = 0.85
    min_overlap_ratio: float = 0.05
    max_distance_helmet_to_vest: float = 200.0

    person_aliases: List[str] = field(default_factory=lambda: ["person", "worker", "employee"])
    helmet_aliases: List[str] = field(default_factory=lambda: ["helmet", "hardhat", "hard_hat", "safety_helmet"])
    vest_aliases: List[str] = field(default_factory=lambda: ["vest", "safety_vest", "reflective_vest", "high_vis_vest"])
    no_helmet_aliases: List[str] = field(default_factory=lambda: ["no_helmet", "no-helmet", "no_hardhat", "no-hardhat"])
    no_vest_aliases: List[str] = field(default_factory=lambda: ["no_vest", "no-vest", "no_safety_vest", "no-safety-vest"])


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _alias_set(values: Iterable[str]) -> set[str]:
    return {normalize_name(v) for v in values}


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersect_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return box_area((x1, y1, x2, y2))


def center_inside(inner: Detection, outer_box: Box) -> bool:
    cx, cy = inner.center()
    x1, y1, x2, y2 = outer_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def overlap_ratio(inner: Detection, outer_box: Box) -> float:
    area = box_area(inner.xyxy)
    if area == 0:
        return 0.0
    return intersect_area(inner.xyxy, outer_box) / area


def make_head_zone(person_box: Box, ratio: float) -> Box:
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return x1, y1, x2, y1 + ratio * h


def make_torso_zone(person_box: Box, y1_ratio: float, y2_ratio: float) -> Box:
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return x1, y1 + y1_ratio * h, x2, y1 + y2_ratio * h


def _best_match(candidates: List[Detection], zone_box: Box, min_overlap_ratio: float) -> Detection | None:
    valid = []
    for det in candidates:
        if center_inside(det, zone_box) or overlap_ratio(det, zone_box) >= min_overlap_ratio:
            valid.append(det)
    if not valid:
        return None
    return max(valid, key=lambda d: d.conf)


def _center_distance(a: Detection, b: Detection) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _same_body_column(helmet: Detection, vest: Detection) -> bool:
    hx1, hy1, hx2, hy2 = helmet.xyxy
    vx1, vy1, vx2, vy2 = vest.xyxy

    # Vest phải nằm dưới helmet.
    if vy1 <= hy1:
        return False

    helmet_w = max(1.0, hx2 - hx1)
    vest_w = max(1.0, vx2 - vx1)

    # Trục x phải chồng nhau đủ rõ, tránh lấy mũ người này ghép với áo người khác.
    overlap_x = max(0.0, min(hx2, vx2) - max(hx1, vx1))
    overlap_ratio_x = overlap_x / min(helmet_w, vest_w)

    if overlap_ratio_x < 0.25:
        return False

    # Khoảng cách dọc không được quá xa.
    vertical_gap = vy1 - hy2
    if vertical_gap > 3.5 * max(helmet_w, 1.0):
        return False

    return True


def analyze_ppe(detections: List[Detection], cfg: RuleConfig | None = None) -> List[PersonPPEStatus]:
    """
    Thuật toán PPE ổn định:

    1. YOLO vẫn detect tất cả class cùng lúc.
    2. Nếu detect được person, lấy person làm gốc rồi ghép helmet/vest vào vùng đầu/thân.
    3. Chỉ dùng fallback helmet+vest để tạo person tạm khi KHÔNG có person nào.
       Việc này tránh đếm thừa người trong ảnh đông công nhân.
    4. Không tạo person tạm từ mỗi helmet đơn lẻ vì dễ gây ảo.
    """
    cfg = cfg or RuleConfig()
    person_names = _alias_set(cfg.person_aliases)
    helmet_names = _alias_set(cfg.helmet_aliases)
    vest_names = _alias_set(cfg.vest_aliases)
    no_helmet_names = _alias_set(cfg.no_helmet_aliases)
    no_vest_names = _alias_set(cfg.no_vest_aliases)

    persons = [d for d in detections if normalize_name(d.cls_name) in person_names]
    helmets = [d for d in detections if normalize_name(d.cls_name) in helmet_names]
    vests = [d for d in detections if normalize_name(d.cls_name) in vest_names]
    no_helmets = [d for d in detections if normalize_name(d.cls_name) in no_helmet_names]
    no_vests = [d for d in detections if normalize_name(d.cls_name) in no_vest_names]

    statuses: List[PersonPPEStatus] = []

    # Case chính: có person thì chỉ phân tích dựa trên person thật.
    # Không tạo fake person từ helmet/vest nữa để tránh Persons bị đếm ảo.
    if persons:
        for person in persons:
            head_zone = make_head_zone(person.xyxy, cfg.head_zone_ratio)
            torso_zone = make_torso_zone(person.xyxy, cfg.torso_y1_ratio, cfg.torso_y2_ratio)

            matched_helmet = _best_match(helmets, head_zone, cfg.min_overlap_ratio)
            matched_vest = _best_match(vests, torso_zone, cfg.min_overlap_ratio)

            explicit_no_helmet = _best_match(no_helmets, head_zone, cfg.min_overlap_ratio) is not None
            explicit_no_vest = _best_match(no_vests, torso_zone, cfg.min_overlap_ratio) is not None

            helmet_ok = matched_helmet is not None and not explicit_no_helmet
            vest_ok = matched_vest is not None and not explicit_no_vest

            violations: List[str] = []
            if not helmet_ok:
                violations.append("NO_HELMET")
            if not vest_ok:
                violations.append("NO_VEST")

            statuses.append(
                PersonPPEStatus(
                    person=person,
                    helmet_ok=helmet_ok,
                    vest_ok=vest_ok,
                    violations=violations,
                    matched_helmet=matched_helmet,
                    matched_vest=matched_vest,
                )
            )

        return statuses

    # Fallback: chỉ khi YOLO không thấy person nào, nhưng thấy helmet + vest rõ ràng gần nhau.
    used_vests: set[int] = set()

    for helmet in helmets:
        best_vest: Detection | None = None
        best_score = -1.0

        for vest in vests:
            if id(vest) in used_vests:
                continue

            if not _same_body_column(helmet, vest):
                continue

            score = helmet.conf + vest.conf - 0.002 * _center_distance(helmet, vest)
            if score > best_score:
                best_score = score
                best_vest = vest

        if best_vest is None:
            continue

        hx1, hy1, hx2, hy2 = helmet.xyxy
        vx1, vy1, vx2, vy2 = best_vest.xyxy

        px1 = min(hx1, vx1)
        py1 = min(hy1, vy1)
        px2 = max(hx2, vx2)
        py2 = max(hy2, vy2)

        fake_person = Detection(
            cls_name="person",
            conf=max(float(helmet.conf), float(best_vest.conf)),
            xyxy=(px1, py1, px2, py2),
            id=getattr(helmet, "id", None),
        )

        used_vests.add(id(best_vest))

        statuses.append(
            PersonPPEStatus(
                person=fake_person,
                helmet_ok=True,
                vest_ok=True,
                violations=[],
                matched_helmet=helmet,
                matched_vest=best_vest,
            )
        )

    return statuses


def summary_counts(statuses: List[PersonPPEStatus]) -> Dict[str, int]:
    total = len(statuses)
    no_helmet = sum("NO_HELMET" in s.violations for s in statuses)
    no_vest = sum("NO_VEST" in s.violations for s in statuses)
    unsafe = sum(bool(s.violations) for s in statuses)
    safe = total - unsafe
    return {
        "persons": total,
        "safe": safe,
        "unsafe": unsafe,
        "no_helmet": no_helmet,
        "no_vest": no_vest,
    }
