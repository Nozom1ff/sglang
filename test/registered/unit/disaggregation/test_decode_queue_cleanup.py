import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _run_as_single_rank(test: unittest.TestCase) -> None:
    # The transfer queue sizes its release collectives from the gloo group;
    # these tests have no process group, so answer as one rank.
    world_size = patch("torch.distributed.get_world_size", return_value=1)
    world_size.start()
    test.addCleanup(world_size.stop)


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        _run_as_single_rank(self)
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_mooncake_failures_hold_destinations_even_without_optional_flag(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        # A restore failure folds into the poll only once transport is terminal
        # (#43562), so Failed is the case that reaches this branch.
        for poll, restore_status in (
            (KVPoll.Failed, HiCacheRestoreResult.READY),
            (KVPoll.Failed, HiCacheRestoreResult.FAILED),
        ):
            with self.subTest(poll=poll, restore_status=restore_status):
                receiver = FakeReceiver()
                receiver.kv_mgr = SimpleNamespace(
                    requires_transfer_drain=True,
                    enable_deferred_decode_kv_release=True,
                )
                receiver.bootstrap_infos = [{"rank": 0}, {"rank": 1}]
                receiver.abort_notified = False
                receiver.abort = MagicMock()

                def ensure_abort_notified(*, force_arm=False, receiver=receiver):
                    receiver.abort()
                    receiver.abort_notified = True

                receiver.ensure_abort_notified = ensure_abort_notified
                req = SimpleNamespace(
                    rid="failed-transfer", bootstrap_room=7, return_logprob=False
                )
                dreq = SimpleNamespace(
                    req=req,
                    kv_receiver=receiver,
                    metadata_buffer_index=3,
                    hicache_restore_status=restore_status,
                )
                q = DecodeTransferQueue.__new__(DecodeTransferQueue)
                q.queue = [dreq]
                q.enable_staging = False
                q.enable_host_receive = False
                q.enable_deferred_kv_release = False
                q.deferred_kv_release_timeout = 30
                q.gloo_group = MagicMock()
                q._deferred_releases = []
                q.tp_rank = 0
                q._poll_with_metadata_gate = MagicMock(return_value=[poll])
                q._clean_hicache_prefetch_resources = MagicMock()
                q.req_to_metadata_buffer_idx_allocator = MagicMock()
                q.scheduler = MagicMock()
                q.scheduler.enable_decode_hicache = False
                q.scheduler.enable_hisparse = True
                q.scheduler.metrics_reporter.enable_metrics = False

                self.assertEqual(q.pop_transferred(), [])
                self.assertEqual(q.queue, [])
                self.assertEqual(len(q._deferred_releases), 1)
                self.assertEqual(q._deferred_releases[0][3:], (3, 2))
                self.assertIs(dreq.kv_receiver, receiver)
                self.assertFalse(receiver.clear_called)
                q._clean_hicache_prefetch_resources.assert_not_called()
                q.scheduler.hisparse_coordinator.request_finished.assert_not_called()
                q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
                mock_release_kv_cache.assert_not_called()
                receiver.abort.assert_called_once()

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler._engine_paused = False
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_releases=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())

        scheduler.disagg_decode_prealloc_queue.retracted_queue.clear()
        self.assertTrue(scheduler.is_fully_idle())
        scheduler.disagg_decode_transfer_queue.has_pending_deferred_releases = lambda: (
            True
        )
        self.assertFalse(scheduler.is_fully_idle())
        # Quarantined buffers have no GPU forward to carry a health response;
        # permit a fresh health request while still blocking cache flush/offload.
        self.assertTrue(scheduler.is_fully_idle(for_health_check=True))


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=SimpleNamespace(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)

    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = [decode_req]
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [decode_req])
        mock_prepare_abort.assert_not_called()
        mock_release_kv_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
