"""In-place compaction must rewind the run originals of a carried repair-merged row (#139210).

A live-replay restore (``repair_alternation=True``) merges a durable ``user;user`` run into one
dict. When a later in-place compaction carries that merged dict into the compacted transcript
(protected head/tail), the merged copy is re-inserted as a fresh row. Archiving the run originals
as ``compacted=1`` leaves them display-visible beside a copy that matches neither of them, so
display dedupe cannot collapse the generations: transcript readers and search see the earlier
user messages twice. The run originals of a carried merged dict are superseded duplicates and
take the same rewind flags (``active=0, compacted=0``) as the carried tail's originals. A merged
dict that is summarized away keeps its run archived as summarized.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB

SUMMARY_TEXT = "## Historical Task Snapshot\nold task\n\n## Goal\ncontinue"


@pytest.fixture()
def db(tmp_path: Path):
    state = SessionDB(db_path=tmp_path / "state.db")
    state.create_session("s1", source="gateway")
    yield state
    state.close()


def _seed(db: SessionDB) -> None:
    rows = [
        ("user", "ask 1"), ("user", "ask 2"),  # durable alternation violation, head region
        ("assistant", "answer 1-2"),
        ("user", "ask 3"), ("user", "ask 4"),  # violation, summarized middle
        ("assistant", "answer 3-4"),
        ("user", "ask 5"),
        ("assistant", "answer 5a"), ("assistant", "answer 5b"),
        ("user", "ask 6"),
        ("assistant", "answer 6"),
    ]
    for i, (role, content) in enumerate(rows):
        db.append_message("s1", role, content, timestamp=1_760_000_000.0 + i * 60)


def _row_flags(db: SessionDB, content: str) -> tuple[int, int]:
    row = db._conn.execute(
        "SELECT active, compacted FROM messages WHERE session_id = ? AND role = 'user' AND content = ?",
        ("s1", content)).fetchone()
    return int(row["active"]), int(row["compacted"])


def _display_user_texts(db: SessionDB) -> list[str]:
    _, display = db.get_resume_conversations("s1")
    return [m["content"] for m in display if m.get("role") == "user"]


def _compress_and_commit(db: SessionDB) -> None:
    """The gateway compaction shape: live-replay restore (no row ids), real compress(), then the
    in-place commit with the exact coverage/tail arguments ``_commit_compaction`` computes."""
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression_archive import coverage_for_commit

    live = db.get_messages_as_conversation("s1", repair_alternation=True)
    assert any("\n\n" in m["content"] for m in live if m.get("role") == "user"), (
        "restore must repair the seeded violation into a merged user row")
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test", quiet_mode=True, protect_first_n=1, protect_last_n=2,
            abort_on_summary_failure=True)
    compressor.bind_session_state(db, "s1")
    with patch("agent.context_compressor.call_llm", return_value=SUMMARY_TEXT):
        compressed = compressor.compress(list(live), force=True)
    assert any(m.get("_compressed_summary") for m in compressed), "compress() must compact here"
    tail_count = sum(1 for m in compressed if isinstance(m, dict) and m.pop("_compaction_tail", None))
    covered_ids, unresolved_held = coverage_for_commit(db, "s1", live)
    db.archive_and_compact(
        "s1", compressed, watermark=db.get_active_message_watermark("s1"),
        tail_count=tail_count, covered_ids=covered_ids, unresolved_held=unresolved_held)


def test_carried_merged_head_run_is_rewound_not_archived(db: SessionDB) -> None:
    _seed(db)
    _compress_and_commit(db)

    # The carried merged head row supersedes its run originals: rewind flags, like the tail's.
    assert _row_flags(db, "ask 1") == (0, 0)
    assert _row_flags(db, "ask 2") == (0, 0)
    # The earlier user texts project exactly once: inside the carried merged copy.
    texts = _display_user_texts(db)
    assert sum("ask 1" in text for text in texts) == 1
    assert sum("ask 2" in text for text in texts) == 1


def test_summarized_merged_run_stays_archived(db: SessionDB) -> None:
    _seed(db)
    _compress_and_commit(db)

    # "ask 3\n\nask 4" was folded into the summary, not carried: its run stays summarized
    # (compacted=1, still display-visible/searchable), never rewound out from under the summary.
    assert _row_flags(db, "ask 3") == (0, 1)
    assert _row_flags(db, "ask 4") == (0, 1)
