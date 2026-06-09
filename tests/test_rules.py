from src.ppe_detector.rules import Detection, analyze_ppe, summary_counts


def test_safe_person_has_helmet_and_vest():
    detections = [
        Detection("person", 0.9, (0, 0, 100, 200)),
        Detection("helmet", 0.8, (30, 5, 70, 40)),
        Detection("safety_vest", 0.85, (20, 70, 80, 160)),
    ]
    statuses = analyze_ppe(detections)
    counts = summary_counts(statuses)
    assert counts["persons"] == 1
    assert counts["safe"] == 1
    assert counts["unsafe"] == 0


def test_missing_vest_is_violation():
    detections = [
        Detection("person", 0.9, (0, 0, 100, 200)),
        Detection("helmet", 0.8, (30, 5, 70, 40)),
    ]
    statuses = analyze_ppe(detections)
    assert statuses[0].violations == ["NO_VEST"]
