"""Runtime contract checks for DistributedTransactionManager compatibility names."""

import pytest

from src.core.distributed_transaction_manager import DistributedTransactionManager


@pytest.mark.asyncio
async def test_legacy_distributed_transaction_methods_support_logical_commit() -> None:
    """Analytics callers should be able to use the legacy DTM method names."""
    dtm = DistributedTransactionManager()

    begin_result = await dtm.begin_distributed_transaction("tx-compat")
    record_result = await dtm.record_operation(
        "tx-compat",
        {"type": "complete_pipeline", "status": "observed"},
    )
    commit_result = await dtm.commit_distributed_transaction("tx-compat")
    final_state = await dtm.get_transaction_state("tx-compat")

    assert begin_result["status"] == "active"
    assert record_result["status"] == "active"
    assert commit_result == {
        "tx_id": "tx-compat",
        "status": "committed",
        "neo4j_committed": False,
        "sqlite_committed": False,
        "operation_count": 1,
        "errors": [],
    }
    assert final_state["status"] == "committed"


@pytest.mark.asyncio
async def test_legacy_distributed_transaction_rollback_uses_current_rollback_path() -> None:
    """Legacy rollback should work for active logical transactions."""
    dtm = DistributedTransactionManager()

    await dtm.begin_distributed_transaction("tx-rollback")
    rollback_result = await dtm.rollback_distributed_transaction("tx-rollback")
    final_state = await dtm.get_transaction_state("tx-rollback")

    assert rollback_result["status"] == "rolled_back"
    assert final_state["status"] == "rolled_back"
