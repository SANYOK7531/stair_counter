"""
Тесты логики TrackHandoff.
Запуск: python tests/test_handoff.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.track_handoff import TrackHandoff


def bbox(cx, cy, w=80, h=160):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def test_simple_handoff():
    """Трек 1 пропал, появился трек 2 рядом → должен получить effective_id=1."""
    h = TrackHandoff(max_age_frames=60, max_distance_px=150)

    # кадр 1: видим трек 1
    mapping = h.resolve({1: bbox(500, 400)}, frame_idx=1)
    assert mapping == {1: 1}

    # кадр 2: трек 1 пропал
    mapping = h.resolve({}, frame_idx=2)
    assert mapping == {}

    # кадр 5: появился трек 2 почти там же
    mapping = h.resolve({2: bbox(510, 395)}, frame_idx=5)
    assert mapping == {2: 1}, f"ожидал effective_id=1, получил {mapping}"
    print("[OK] simple handoff")


def test_no_handoff_if_too_far():
    h = TrackHandoff(max_distance_px=100)
    h.resolve({1: bbox(100, 100)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # появился трек очень далеко
    mapping = h.resolve({2: bbox(800, 600)}, frame_idx=3)
    assert mapping == {2: 2}, "далёкий трек не должен наследовать"
    print("[OK] no handoff if too far")


def test_no_handoff_if_too_old():
    h = TrackHandoff(max_age_frames=10)
    h.resolve({1: bbox(500, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # появился трек через 50 кадров — слишком поздно
    mapping = h.resolve({2: bbox(505, 405)}, frame_idx=52)
    assert mapping == {2: 2}, "старый пропавший не должен быть родителем"
    print("[OK] no handoff if too old")


def test_no_handoff_if_size_mismatch():
    h = TrackHandoff(size_ratio_tolerance=0.5)
    # трек 1: большой
    h.resolve({1: bbox(500, 400, w=160, h=320)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # трек 2: крошечный рядом — это явно другой объект
    mapping = h.resolve({2: bbox(505, 405, w=20, h=40)}, frame_idx=3)
    assert mapping == {2: 2}
    print("[OK] no handoff if size mismatch")


def test_parent_used_only_once():
    """Если пропал один, а появилось двое — только один получит его ID."""
    h = TrackHandoff()
    h.resolve({1: bbox(500, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # двое появились рядом одновременно
    mapping = h.resolve(
        {2: bbox(505, 395), 3: bbox(510, 405)},
        frame_idx=3,
    )
    parent_inheritors = [tid for tid, eff in mapping.items() if eff == 1]
    assert len(parent_inheritors) == 1, (
        f"только один должен унаследовать, а получили: {mapping}"
    )
    print("[OK] parent used only once")


def test_closest_wins():
    """Из двух пропавших родителем должен стать ближайший."""
    h = TrackHandoff()
    # трек 1 слева, трек 2 справа
    h.resolve({1: bbox(100, 400), 2: bbox(700, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # новый появился ближе к 2
    mapping = h.resolve({99: bbox(690, 400)}, frame_idx=3)
    assert mapping == {99: 2}, f"должен был унаследовать 2, получили {mapping}"
    print("[OK] closest parent wins")


def test_existing_track_not_touched():
    """Продолжающийся трек не должен "усыновляться" другим."""
    h = TrackHandoff()
    h.resolve({1: bbox(500, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)  # 1 "пропал"
    # 1 снова появился — BoT-SORT дал ему тот же ID (не новый)
    # но в handoff мы должны увидеть что он тот же, а не сделать новый матчинг
    # (в нашей логике это естественно: 1 не является "новым" треком)
    mapping = h.resolve({1: bbox(505, 405)}, frame_idx=3)
    assert mapping == {1: 1}, "вернувшийся тот же ID должен оставаться собой"
    print("[OK] same id returning stays itself")


def test_chain_handoff_via_root():
    """Если 1 → 2 (через handoff), а потом 2 пропал и появился 3 — 3 должен указывать на 1."""
    h = TrackHandoff()
    h.resolve({1: bbox(500, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    # трек 2 унаследовал 1
    mapping = h.resolve({2: bbox(505, 405)}, frame_idx=3)
    assert mapping == {2: 1}
    # теперь трек 2 исчез
    h.resolve({}, frame_idx=4)
    # появился трек 3 рядом
    mapping = h.resolve({3: bbox(510, 410)}, frame_idx=5)
    assert mapping == {3: 1}, (
        f"цепочка должна вести к корню 1, получили {mapping}"
    )
    print("[OK] chain handoff resolves to root")


def test_forget_clears_state():
    h = TrackHandoff()
    h.resolve({1: bbox(500, 400)}, frame_idx=1)
    h.resolve({}, frame_idx=2)
    h.forget({1})
    # теперь появившийся рядом не должен найти родителя
    mapping = h.resolve({2: bbox(505, 405)}, frame_idx=3)
    assert mapping == {2: 2}
    print("[OK] forget clears state")


def test_velocity_prediction_tracks_moving_person():
    """
    Человек шёл вправо со скоростью 20 px/кадр. Пропал. Новый трек
    появился в предсказанной точке — должен склеиться, даже если это
    далеко от последней известной позиции.
    """
    h = TrackHandoff(
        max_age_frames=30,
        max_distance_px=80,   # строгое ограничение!
        max_predicted_speed_px=100.0,
    )
    # два кадра движения вправо — набираем скорость
    h.resolve({1: bbox(400, 300)}, frame_idx=1)
    h.resolve({1: bbox(420, 300)}, frame_idx=2)  # +20 по x
    h.resolve({1: bbox(440, 300)}, frame_idx=3)  # +20 по x
    # пропал
    h.resolve({}, frame_idx=4)
    # через 10 кадров появился новый где он ДОЛЖЕН быть с учётом движения:
    # последняя позиция 440, скорость 20 px/кадр, age=10 → предсказанная 440+200=640
    mapping = h.resolve({99: bbox(635, 300)}, frame_idx=14)
    assert mapping == {99: 1}, (
        f"должен склеиться с предсказанной позиции, получили {mapping}"
    )
    print("[OK] velocity prediction finds moving person")


def test_velocity_prediction_rejects_opposite_direction():
    """
    Человек шёл вправо. Новый трек появился в противоположной стороне —
    не склеиваем, это другой человек идущий в другую сторону.
    """
    h = TrackHandoff(max_distance_px=80, max_predicted_speed_px=100.0)
    h.resolve({1: bbox(400, 300)}, frame_idx=1)
    h.resolve({1: bbox(420, 300)}, frame_idx=2)
    h.resolve({1: bbox(440, 300)}, frame_idx=3)
    h.resolve({}, frame_idx=4)
    # через 10 кадров появился слева от последней позиции — противоположно движению
    mapping = h.resolve({99: bbox(350, 300)}, frame_idx=14)
    assert mapping == {99: 99}, (
        f"движение в противоположную сторону — не должен матчиться, получили {mapping}"
    )
    print("[OK] opposite direction rejected")


if __name__ == "__main__":
    test_simple_handoff()
    test_no_handoff_if_too_far()
    test_no_handoff_if_too_old()
    test_no_handoff_if_size_mismatch()
    test_parent_used_only_once()
    test_closest_wins()
    test_existing_track_not_touched()
    test_chain_handoff_via_root()
    test_forget_clears_state()
    test_velocity_prediction_tracks_moving_person()
    test_velocity_prediction_rejects_opposite_direction()
    print("\nвсе тесты прошли ✓")
