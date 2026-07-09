"""Session radix cache for UnifiedRadixCache (``--enable-session-radix-cache``):
tag each request's KV path by session_id and slide per-session SWA / Mamba
component references; ``release_radix_session`` (close) dereferences a session's
tagged nodes. Independent of the RadixCache-era ``SessionRadixCacheMixin``."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Set

from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit open_radix_session clears the tombstone for intentional reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192

TIER_UNUSED = 0
TIER_REF = 1


def _classify_node_tier(node: UnifiedTreeNode) -> int:
    return TIER_REF if node.session_ref > 0 else TIER_UNUSED


# Imported after the tier constants: unified_cache_components.__init__ pulls in
# full_component, which imports TIER_* / _classify_node_tier back from here, so
# those names must be bound before this import can re-enter this module.
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)


class SessionUnifiedRadixCacheMixin:
    """Tags unified radix KV by session id; a node holds the set of sessions on
    it, and per-session it also tracks the current SWA window segment and the
    Mamba frontier node. ``release_radix_session`` (close) releases a session's
    reference; tagged KV is evicted with reference counters -- no pinning, no
    open. Mixed into UnifiedRadixCache."""

    def _reset_session_radix_state(self) -> None:
        self.session_id_to_ref_nodes: Dict[str, Set[UnifiedTreeNode]] = {}
        self._closed_session_ids: OrderedDict = OrderedDict()
        self._session_swa_window_nodes: Dict[str, Set[UnifiedTreeNode]] = {}
        self._session_mamba_frontier: Dict[str, UnifiedTreeNode] = {}
        # Parallel tier views over evictable_device_leaves; disjoint and their
        # union == evictable_device_leaves. Always created (harmless when the
        # flag is off) so forget/sanity_check can reference them unconditionally.
        self.unused_evictable_device_leaves: Set[UnifiedTreeNode] = set()
        self.referenced_evictable_device_leaves: Set[UnifiedTreeNode] = set()
        self.session_ref_evictions: Dict[ComponentType, int] = {
            ct: 0 for ct in self.tree_components
        }

    def session_id_for_req(self, req: Req) -> Optional[str]:
        if req.session_id is not None:
            return req.session_id
        if req.session is not None:
            return req.session.session_id
        return None

    def register_session_ref(self, req: Req) -> None:
        if not self.enable_session_radix_cache:
            return
        if req.session is not None and req.session.streaming:
            return
        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        ref_nodes = self.session_id_to_ref_nodes.get(session_id)
        tracked_nodes = ref_nodes if ref_nodes is not None else set()

        last_node = req.last_node
        if last_node not in (None, self.root_node):
            frontier = last_node
            new_nodes = self._collect_untracked_nodes_from_last_node(
                last_node, tracked_nodes
            )
        else:
            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return
            radix_key = RadixKey(token_ids, req.extra_key).page_aligned(self.page_size)
            if len(radix_key) == 0:
                return
            nodes_on_path = self._collect_nodes_on_path(radix_key)
            frontier = nodes_on_path[-1] if nodes_on_path else None
            new_nodes = [node for node in nodes_on_path if node not in tracked_nodes]

        if new_nodes:
            if ref_nodes is None:
                ref_nodes = self.session_id_to_ref_nodes.setdefault(session_id, set())
            for node in new_nodes:
                self._inc_session_ref(node)
                ref_nodes.add(node)
                if node.tracked_session_ids is None:
                    node.tracked_session_ids = set()
                node.tracked_session_ids.add(session_id)

        # Slide after session_ref accounting so invariant 6 holds; frontier None
        # (slow path with no match) means nothing to slide.
        if frontier is not None:
            if self.supports_swa():
                self._slide_swa_window(session_id, frontier)
            if self.supports_mamba():
                self._slide_mamba_frontier(session_id, frontier)

    def _collect_nodes_on_path(self, key: RadixKey) -> list:
        node = self.root_node
        nodes = []
        while len(key) > 0:
            child_key = key.child_key(self.page_size)
            # node.children may be a defaultdict (fresh nodes); .get avoids
            # fabricating a child on a bare index read.
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len <= 0:
                break
            nodes.append(child)
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
        return nodes

    def _collect_untracked_nodes_from_last_node(
        self, node: Optional[UnifiedTreeNode], tracked_nodes: Set[UnifiedTreeNode]
    ) -> list:
        nodes = []
        while node not in (None, self.root_node):
            if node in tracked_nodes:
                break
            nodes.append(node)
            node = node.parent
        return nodes

    def _slide_swa_window(self, session_id: str, frontier: UnifiedTreeNode) -> None:
        old = self._session_swa_window_nodes.get(session_id, set())
        new = set()
        span = self.components[ComponentType.SWA].sliding_window_size + self.page_size
        node, acc = frontier, 0
        while node not in (None, self.root_node) and acc < span:
            new.add(node)
            acc += len(node.key)
            node = node.parent
        for n in old - new:
            self._dec_swa_window_ref(n)
        for n in new - old:
            self._inc_swa_window_ref(n)
        if new:
            self._session_swa_window_nodes[session_id] = new
        else:
            self._session_swa_window_nodes.pop(session_id, None)

    def _slide_mamba_frontier(self, session_id: str, frontier: UnifiedTreeNode) -> None:
        old = self._session_mamba_frontier.get(session_id)
        if old is frontier:
            return
        if old is not None:
            self._dec_mamba_frontier_ref(old)
        self._inc_mamba_frontier_ref(frontier)
        self._session_mamba_frontier[session_id] = frontier

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> None:
        if not self.enable_session_radix_cache:
            return
        self._closed_session_ids.pop(session_id, None)

    def release_radix_session(self, session_id: str) -> int:
        # Only release reference instead of evicting session's KV now.
        if not self.enable_session_radix_cache or session_id is None:
            return 0
        self._remember_closed_session(session_id)

        window = self._session_swa_window_nodes.pop(session_id, None)
        if window is not None:
            for node in window:
                self._dec_swa_window_ref(node)
        frontier = self._session_mamba_frontier.pop(session_id, None)
        if frontier is not None:
            self._dec_mamba_frontier_ref(frontier)

        ref_nodes = self.session_id_to_ref_nodes.pop(session_id, None)
        if not ref_nodes:
            return 0
        for node in ref_nodes:
            self._dec_session_ref(node)
            if node.tracked_session_ids is not None:
                node.tracked_session_ids.discard(session_id)
                if not node.tracked_session_ids:
                    node.tracked_session_ids = None
        logger.info(
            "release_radix_session %s: dereferenced %d nodes",
            session_id,
            len(ref_nodes),
        )
        return len(ref_nodes)

    def _update_session_leaf_tier(
        self, node: UnifiedTreeNode, is_device_leaf: bool
    ) -> None:
        """Maintain the parallel unused/referenced tier views. O(1) set ops:
        no tree walking (called from the hot choke point)."""
        if is_device_leaf:
            if _classify_node_tier(node) == TIER_REF:
                self.referenced_evictable_device_leaves.add(node)
                self.unused_evictable_device_leaves.discard(node)
            else:
                self.unused_evictable_device_leaves.add(node)
                self.referenced_evictable_device_leaves.discard(node)
        else:
            self.unused_evictable_device_leaves.discard(node)
            self.referenced_evictable_device_leaves.discard(node)

    def _inc_session_ref(self, node: UnifiedTreeNode) -> None:
        node.session_ref += 1
        self._update_evictable_leaf_sets(node)

    def _dec_session_ref(self, node: UnifiedTreeNode) -> None:
        node.session_ref = max(0, node.session_ref - 1)
        self._update_evictable_leaf_sets(node)

    def _inc_swa_window_ref(self, node: UnifiedTreeNode) -> None:
        node.swa_window_ref += 1

    def _dec_swa_window_ref(self, node: UnifiedTreeNode) -> None:
        node.swa_window_ref = max(0, node.swa_window_ref - 1)
        # Retired segment: MATCH_END just refreshed it to MRU; sink it to the
        # LRU tail so it becomes the first eviction candidate without freeing.
        if node.swa_window_ref == 0:
            lru = self.lru_lists.get(ComponentType.SWA)
            if lru is not None and lru.in_list(node):
                lru.demote_to_lru(node)

    def _inc_mamba_frontier_ref(self, node: UnifiedTreeNode) -> None:
        node.mamba_frontier_ref += 1

    def _dec_mamba_frontier_ref(self, node: UnifiedTreeNode) -> None:
        node.mamba_frontier_ref = max(0, node.mamba_frontier_ref - 1)
        if node.mamba_frontier_ref == 0:
            lru = self.lru_lists.get(ComponentType.MAMBA)
            if lru is not None and lru.in_list(node):
                lru.demote_to_lru(node)

    def _session_on_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        if not self.enable_session_radix_cache:
            return
        # Both halves stay in the protected span; mamba state stays on child so
        # the frontier pointer does not move.
        new_parent.session_ref = child.session_ref
        new_parent.swa_window_ref = child.swa_window_ref
        if child.tracked_session_ids:
            new_parent.tracked_session_ids = set(child.tracked_session_ids)
            for session_id in new_parent.tracked_session_ids:
                nodes = self.session_id_to_ref_nodes.get(session_id)
                if nodes is not None:
                    nodes.add(new_parent)
                window = self._session_swa_window_nodes.get(session_id)
                if window is not None and child in window:
                    window.add(new_parent)

    def _session_forget_node(self, node: UnifiedTreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        if node.tracked_session_ids:
            for session_id in node.tracked_session_ids:
                nodes = self.session_id_to_ref_nodes.get(session_id)
                if nodes is not None:
                    nodes.discard(node)
                window = self._session_swa_window_nodes.get(session_id)
                if window is not None:
                    window.discard(node)
                if self._session_mamba_frontier.get(session_id) is node:
                    del self._session_mamba_frontier[session_id]
            node.tracked_session_ids = None
        self.unused_evictable_device_leaves.discard(node)
        self.referenced_evictable_device_leaves.discard(node)
