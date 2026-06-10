"""Tests for emeg_fm.stimuli.ImageStimulusSet — image discovery, marker-code
assignment, presentation schedule, and gallery (de)serialization. CLIP
embedding is not exercised (it needs torch + a model download); the gallery
roundtrip is tested with an injected array.
"""
import numpy as np
import pytest

from emeg_fm.stimuli import ImageStimulusSet


def _make_images(d, names):
    for n in names:
        (d / n).write_bytes(b"\xff\xd8\xff")     # not a real jpeg; from_dir only lists
    return d


def test_from_dir_assigns_sorted_1indexed_codes(tmp_path):
    _make_images(tmp_path, ["c.jpg", "a.jpg", "b.png"])
    ss = ImageStimulusSet.from_dir(tmp_path)
    assert list(ss.codes) == [1, 2, 3]
    assert [p.rsplit("/", 1)[-1] for p in ss.paths] == ["a.jpg", "b.png", "c.jpg"]
    assert ss.code_to_basename == {1: "a.jpg", 2: "b.png", 3: "c.jpg"}


def test_from_dir_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageStimulusSet.from_dir(tmp_path)


def test_from_dir_max_images_reproducible(tmp_path):
    _make_images(tmp_path, [f"img{i:02d}.jpg" for i in range(20)])
    a = ImageStimulusSet.from_dir(tmp_path, max_images=5, seed=1)
    b = ImageStimulusSet.from_dir(tmp_path, max_images=5, seed=1)
    assert len(a.paths) == 5
    assert a.paths == b.paths


def test_build_schedule_counts_and_codes(tmp_path):
    _make_images(tmp_path, [f"i{i}.jpg" for i in range(4)])
    ss = ImageStimulusSet.from_dir(tmp_path)
    sched = ss.build_schedule(n_repeats=3, seed=0)
    assert len(sched) == 12
    codes = [c for c, _ in sched]
    # every image code appears exactly n_repeats times
    for code in ss.codes:
        assert codes.count(int(code)) == 3
    # all paths resolve through the code map
    cmap = ss.code_to_path
    assert all(p == cmap[c] for c, p in sched)


def test_build_schedule_deterministic(tmp_path):
    _make_images(tmp_path, [f"i{i}.jpg" for i in range(5)])
    ss = ImageStimulusSet.from_dir(tmp_path)
    assert ss.build_schedule(4, seed=7) == ss.build_schedule(4, seed=7)


def test_save_schedule_roundtrip(tmp_path):
    import json
    _make_images(tmp_path, [f"i{i}.jpg" for i in range(3)])
    ss = ImageStimulusSet.from_dir(tmp_path)
    out = tmp_path / "sched.json"
    ss.save_schedule(out, n_repeats=2, seed=0)
    payload = json.loads(out.read_text())
    assert payload["n_repeats"] == 2
    assert len(payload["trials"]) == 6


def test_gallery_save_load_roundtrip(tmp_path):
    _make_images(tmp_path, [f"i{i}.jpg" for i in range(4)])
    ss = ImageStimulusSet.from_dir(tmp_path)
    ss.gallery = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)
    ss.gallery_ids = ss.codes.copy()
    ss.clip_model = "fake/clip"
    out = tmp_path / "gal.npz"
    ss.save_gallery(out)
    loaded = ImageStimulusSet.load_gallery(out)
    np.testing.assert_array_equal(loaded.gallery, ss.gallery)
    np.testing.assert_array_equal(loaded.gallery_ids, ss.gallery_ids)
    assert loaded.clip_model == "fake/clip"
    assert len(loaded.paths) == 4
