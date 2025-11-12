import pytest

from src.database import PlayerDB


@pytest.fixture()
def temp_player_db(tmp_path):
    db_path = tmp_path / "players.db"
    db = PlayerDB(db_path=str(db_path))
    try:
        yield db
    finally:
        db.close()
        if db_path.exists():
            db_path.unlink()


def _create_user(db: PlayerDB, email: str = "test@example.com") -> int:
    return db.create_user(
        email=email,
        password_hash="hash",
        first_name="Test",
        last_name="User",
        is_admin=False,
    )


def test_upsert_and_get_journal_entry(temp_player_db: PlayerDB):
    user_id = _create_user(temp_player_db)

    temp_player_db.upsert_journal_entry(
        user_id=user_id,
        entry_date="2025-01-01",
        visibility="private",
        title="Day 1",
        body="Initial notes",
    )

    entry = temp_player_db.get_journal_entry(user_id, "2025-01-01", "private")
    assert entry is not None
    assert entry["title"] == "Day 1"
    assert entry["body"] == "Initial notes"
    assert entry["visibility"] == "private"

    # Update existing entry
    temp_player_db.upsert_journal_entry(
        user_id=user_id,
        entry_date="2025-01-01",
        visibility="PRIVATE",  # ensure visibility normalization
        title="Updated Day 1",
        body="Revised notes",
    )

    updated_entry = temp_player_db.get_journal_entry(user_id, "2025-01-01", "private")
    assert updated_entry["title"] == "Updated Day 1"
    assert updated_entry["body"] == "Revised notes"


def test_list_and_delete_journal_entries(temp_player_db: PlayerDB):
    user_id = _create_user(temp_player_db, email="player@example.com")

    entries = [
        ("2025-03-01", "public", "Game day", "Felt locked in"),
        ("2025-03-02", "private", "Work", "Focused on mechanics"),
        ("2025-03-03", "public", "Recovery", "Active recovery session"),
    ]
    for date_value, visibility, title, body in entries:
        temp_player_db.upsert_journal_entry(
            user_id=user_id,
            entry_date=date_value,
            visibility=visibility,
            title=title,
            body=body,
        )

    all_entries = temp_player_db.list_journal_entries(user_id)
    assert len(all_entries) == 3
    # Entries are ordered newest first
    assert [item["entry_date"] for item in all_entries] == ["2025-03-03", "2025-03-02", "2025-03-01"]

    public_entries = temp_player_db.list_journal_entries(user_id, visibility="public")
    assert len(public_entries) == 2
    assert all(entry["visibility"] == "public" for entry in public_entries)

    filtered_entries = temp_player_db.list_journal_entries(
        user_id,
        start_date="2025-03-02",
        end_date="2025-03-03",
    )
    assert len(filtered_entries) == 2
    assert {entry["entry_date"] for entry in filtered_entries} == {"2025-03-02", "2025-03-03"}

    entry_to_delete = public_entries[0]
    assert temp_player_db.delete_journal_entry(entry_to_delete["id"], user_id) is True
    assert temp_player_db.get_journal_entry(user_id, entry_to_delete["entry_date"], entry_to_delete["visibility"]) is None


