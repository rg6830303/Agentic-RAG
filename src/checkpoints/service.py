from __future__ import annotations

from src.docstore.sqlite_store import SQLiteDocstore
from src.utils.hashing import checksum_text
from src.utils.models import CheckpointRecord, CheckpointStatus
from src.utils.time import utc_now_iso


class CheckpointManager:
    def __init__(self, docstore: SQLiteDocstore) -> None:
        self.docstore = docstore

    def create(
        self,
        stage: str,
        payload: dict[str, object],
        enabled: bool,
        requires_human: bool = False,
        notes: str | None = None,
    ) -> CheckpointRecord:
        created_at = utc_now_iso()
        status = (
            CheckpointStatus.PENDING.value
            if enabled and requires_human
            else CheckpointStatus.AUTO_APPROVED.value
        )
        record = CheckpointRecord(
            checkpoint_id=checksum_text(f"{stage}:{created_at}:{payload}")[:24],
            stage=stage,
            status=status,
            created_at=created_at,
            payload=payload,
            notes=notes,
        )
        self.docstore.persist_checkpoint(record)
        return record

    def approve(self, checkpoint_id: str, notes: str | None = None) -> None:
        self.docstore.update_checkpoint_status(
            checkpoint_id=checkpoint_id,
            status=CheckpointStatus.APPROVED.value,
            notes=notes,
        )

    def reject(self, checkpoint_id: str, notes: str | None = None) -> None:
        self.docstore.update_checkpoint_status(
            checkpoint_id=checkpoint_id,
            status=CheckpointStatus.REJECTED.value,
            notes=notes,
        )
