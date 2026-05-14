"""
Юнит-тесты логики DirectionalZoneCounter.
Запуск: python tests/test_counter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.directional_zone_counter import DirectionalZoneCounter


# зоны 100x100
ENTRY = [(0, 100), (100, 100), (100, 200), (0, 200)]   # нижний квадрат в кадре
EXIT = [(0, 0), (100, 0), (100, 100), (0, 100)]        # верхний квадрат


def make(min_coverage: float = 0.0):
    # по умолчанию отключаем проверку покрытия (0.0), кроме специальных тестов
    return DirectionalZoneCounter(
        "test", ENTRY, EXIT,
        min_frames_in_zone=3,
        min_exit_coverage=min_coverage,
    )


def foot_bbox_at(x: int, y: int) -> tuple[int, int, int, int]:
    """bbox 80x80 такой, что нога (y2) попадает в (x, y)."""
    return (x - 40, y - 80, x + 40, y)


# ── базовая логика последовательности ──────────────────────────────────────

def test_normal_climb_counts_one():
    c = make()
    for _ in range(3):
        c.update(1, foot_bbox_at(50, 150))  # нога в ENTRY
    for _ in range(3):
        c.update(1, foot_bbox_at(50, 50))   # нога в EXIT
    assert c.total_count == 1, c.total_count
    print("[OK] normal climb -> +1")


def test_only_exit_does_not_count():
    c = make()
    for _ in range(10):
        c.update(2, foot_bbox_at(50, 50))
    assert c.total_count == 0, c.total_count
    print("[OK] only-exit -> 0")


def test_reverse_direction_does_not_count():
    c = make()
    for _ in range(3):
        c.update(3, foot_bbox_at(50, 50))
    for _ in range(3):
        c.update(3, foot_bbox_at(50, 150))
    assert c.total_count == 0, c.total_count
    print("[OK] exit -> entry -> 0")


def test_brief_flicker_ignored():
    c = make()
    c.update(4, foot_bbox_at(50, 150))   # 1 кадр в ENTRY — мало
    c.update(4, foot_bbox_at(200, 200))  # вне зон
    for _ in range(3):
        c.update(4, foot_bbox_at(50, 50))  # 3 в EXIT
    assert c.total_count == 0, c.total_count
    print("[OK] brief flicker -> 0")


def test_climb_then_descend_counted_once():
    c = make()
    for _ in range(3):
        c.update(5, foot_bbox_at(50, 150))
    for _ in range(3):
        c.update(5, foot_bbox_at(50, 50))
    for _ in range(3):
        c.update(5, foot_bbox_at(50, 150))
    assert c.total_count == 1, c.total_count
    print("[OK] climb + descend -> 1")


def test_two_people_independent():
    c = make()
    for _ in range(3):
        c.update(10, foot_bbox_at(50, 150))
    for _ in range(3):
        c.update(10, foot_bbox_at(50, 50))
    for _ in range(5):
        c.update(11, foot_bbox_at(50, 50))
    assert c.total_count == 1, c.total_count
    print("[OK] two people independent")


def test_cleanup_forgets_track():
    c = make()
    for _ in range(3):
        c.update(20, foot_bbox_at(50, 150))
    c.cleanup({20})
    for _ in range(3):
        c.update(20, foot_bbox_at(50, 50))
    assert c.total_count == 0, c.total_count
    print("[OK] cleanup wipes history")


# ── coverage-логика ────────────────────────────────────────────────────────

def test_exit_coverage_blocks_small_bbox():
    """
    Порог coverage = 50%. Крошечный bbox в EXIT даёт coverage ≈ 0.01 — не считаем.
    """
    c = make(min_coverage=0.5)
    for _ in range(3):
        c.update(30, foot_bbox_at(50, 150))  # честно прошли ENTRY
    # крошечный bbox (10x10) с ногой в EXIT, но покрытие ~1% EXIT
    tiny_bbox = (45, 45, 55, 55)  # y2=55 в EXIT
    for _ in range(10):
        c.update(30, tiny_bbox)
    assert c.total_count == 0, (
        f"ожидал 0 (coverage < 0.5), получил {c.total_count}"
    )
    print("[OK] small bbox in EXIT blocked by coverage")


def test_exit_coverage_accepts_big_bbox():
    """Порог 0.3, большой bbox покрывает EXIT целиком — засчитывается."""
    c = make(min_coverage=0.3)
    for _ in range(3):
        c.update(31, foot_bbox_at(50, 150))
    big_bbox = (-10, -10, 110, 90)  # нога y2=90 в EXIT, покрытие ~90%
    for _ in range(3):
        c.update(31, big_bbox)
    assert c.total_count == 1, c.total_count
    print("[OK] big bbox in EXIT accepted")


def test_exit_coverage_zero_disables_check():
    """min_coverage=0 — проверка выключена, работает как раньше."""
    c = make(min_coverage=0.0)
    for _ in range(3):
        c.update(32, foot_bbox_at(50, 150))
    tiny = (48, 48, 52, 52)  # нога y2=52 в EXIT
    for _ in range(3):
        c.update(32, tiny)
    assert c.total_count == 1, c.total_count
    print("[OK] coverage=0 disables check")


def test_entry_does_not_care_about_coverage():
    """Coverage проверяется только для EXIT — ENTRY пропускает мелких."""
    c = make(min_coverage=0.9)  # очень жёсткий порог для EXIT
    # крошечный bbox в ENTRY — должен зарегиться (coverage ENTRY не смотрим)
    tiny_in_entry = (45, 145, 55, 155)
    for _ in range(3):
        c.update(33, tiny_in_entry)
    # большой bbox в EXIT — coverage ~90%, пройдёт порог 0.9
    big_in_exit = (0, 0, 100, 90)
    for _ in range(3):
        c.update(33, big_in_exit)
    assert c.total_count == 1, c.total_count
    print("[OK] ENTRY ignores coverage, EXIT enforces it")


if __name__ == "__main__":
    test_normal_climb_counts_one()
    test_only_exit_does_not_count()
    test_reverse_direction_does_not_count()
    test_brief_flicker_ignored()
    test_climb_then_descend_counted_once()
    test_two_people_independent()
    test_cleanup_forgets_track()
    test_exit_coverage_blocks_small_bbox()
    test_exit_coverage_accepts_big_bbox()
    test_exit_coverage_zero_disables_check()
    test_entry_does_not_care_about_coverage()
    print("\nвсе тесты прошли ✓")
