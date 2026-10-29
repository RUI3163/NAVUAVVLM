"""VLM worker: priority-queued client + separate OS subprocess.

Priorities: 1 arrival verification > 2 grounding > 3 captioning. One request
in flight at a time (4 GB VRAM). Stale / superseded results are dropped.
Any backend failure (crash, timeout, missing weights) degrades to the
deterministic rule backend so the mission never depends on the VLM.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import numpy as np

from vlnav.parser import ground_to_node, parse_instruction, verify_intersection_rule

log = logging.getLogger("vlnav.vlm")

ARRIVAL_PROMPT = "Is the drone above or at a road intersection? Answer YES or NO."


class VLMKind(IntEnum):
    ARRIVAL = 1
    GROUNDING = 2
    CAPTION = 3


@dataclass
class VLMRequest:
    req_id: int
    kind: VLMKind
    payload: dict
    created: float
    ttl_s: float
    key: str


@dataclass
class VLMResult:
    req_id: int
    kind: VLMKind
    answer: Any
    latency_s: float
    backend: str


@dataclass
class VLMConfig:
    mode: str = "rule"               # "rule" | "subprocess"
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_pixels: int = 448 * 448      # keep VRAM small
    request_timeout_s: float = 20.0  # subprocess round-trip before fallback
    ttl_s: dict = field(default_factory=lambda: {1: 6.0, 2: 15.0, 3: 30.0})


# ---------------------------------------------------------------- backends
class RuleBackend:
    name = "rule"

    def run(self, req: VLMRequest) -> Any:
        if req.kind is VLMKind.ARRIVAL:
            return "YES" if verify_intersection_rule(req.payload["depth"]) else "NO"
        if req.kind is VLMKind.GROUNDING:
            p = parse_instruction(req.payload["text"])
            return ground_to_node(p, req.payload["graph"], req.payload["start_node"]).node
        return "road segment"  # caption placeholder


class QwenBackend:
    """Qwen2-VL-2B, 4-bit (bitsandbytes). Imports are lazy: runs only in the subprocess."""
    name = "qwen"

    def __init__(self, cfg: VLMConfig) -> None:
        import torch
        from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model_id, quantization_config=q, device_map="auto", torch_dtype=torch.float16
        ).eval()
        self.proc = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
        self._torch = torch

    def _ask(self, rgb: np.ndarray, prompt: str, max_new: int) -> str:
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[Image.fromarray(rgb)], return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def run(self, req: VLMRequest) -> Any:
        if req.kind is VLMKind.ARRIVAL:
            ans = self._ask(req.payload["rgb"], ARRIVAL_PROMPT, 3).upper()
            return "YES" if ans.startswith("YES") else "NO"
        if req.kind is VLMKind.CAPTION:
            return self._ask(req.payload["rgb"], "Describe this aerial road view in one short sentence.", 40)
        return RuleBackend().run(req)  # grounding uses the deterministic parser (graph isn't picklable-cheap)


def worker_main(req_q, res_q, cfg: VLMConfig) -> None:
    try:
        backend = QwenBackend(cfg)
    except Exception as e:  # noqa: BLE001
        res_q.put(("fatal", repr(e)))
        return
    res_q.put(("ready", backend.name))
    while True:
        req = req_q.get()
        if req is None:
            return
        t0 = time.monotonic()
        try:
            ans = backend.run(req)
        except Exception as e:  # noqa: BLE001
            res_q.put(("err", req.req_id, repr(e)))
            continue
        res_q.put(("ok", VLMResult(req.req_id, req.kind, ans, time.monotonic() - t0, backend.name)))


# ------------------------------------------------------------------ client
class VLMClient:
    def __init__(self, cfg: VLMConfig, clock=time.monotonic) -> None:
        self.cfg, self._clock = cfg, clock
        self._rule = RuleBackend()
        self._heap: list[tuple[int, int, VLMRequest]] = []
        self._seq = itertools.count(1)
        self._cv = threading.Condition()
        self._latest_key: dict[str, int] = {}
        self._results: "queue.Queue[tuple[VLMRequest, VLMResult]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._req_q = self._res_q = None
        self._sub_ok = False
        self.latencies: list[float] = []

    # lifecycle
    def start(self) -> None:
        if self.cfg.mode == "subprocess":
            ctx = mp.get_context("spawn")  # CUDA-safe, isolated GPU context
            self._req_q, self._res_q = ctx.Queue(maxsize=2), ctx.Queue()
            self._proc = ctx.Process(target=worker_main, args=(self._req_q, self._res_q, self.cfg), daemon=True)
            self._proc.start()
        self._thread = threading.Thread(target=self._dispatch, name="vlm-dispatch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._req_q is not None:
            try:
                self._req_q.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
        if self._proc is not None:
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.terminate()

    # API (non-blocking)
    def submit(self, kind: VLMKind, payload: dict, key: Optional[str] = None, ttl_s: Optional[float] = None) -> int:
        rid = next(self._seq)
        key = key or kind.name
        req = VLMRequest(rid, kind, payload, self._clock(), ttl_s or self.cfg.ttl_s[int(kind)], key)
        with self._cv:
            self._latest_key[key] = rid  # supersedes older same-key requests
            heapq.heappush(self._heap, (int(kind), rid, req))
            self._cv.notify()
        return rid

    def poll(self) -> list[VLMResult]:
        out = []
        while True:
            try:
                req, res = self._results.get_nowait()
            except queue.Empty:
                return out
            if self._clock() - req.created > req.ttl_s or self._latest_key.get(req.key) != req.req_id:
                log.info("drop stale VLM result %s", req.kind.name)
                continue
            out.append(res)

    def fallback_verify(self, depth: np.ndarray) -> str:
        return "YES" if verify_intersection_rule(depth) else "NO"

    # internals
    def _dispatch(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._heap and not self._stop.is_set():
                    self._cv.wait(0.2)
                if self._stop.is_set():
                    return
                _, _, req = heapq.heappop(self._heap)
            if self._clock() - req.created > req.ttl_s or self._latest_key.get(req.key) != req.req_id:
                continue  # stale before even running
            res = self._execute(req)
            self.latencies.append(res.latency_s)
            self._results.put((req, res))

    def _execute(self, req: VLMRequest) -> VLMResult:
        if self._proc is not None and req.kind is not VLMKind.GROUNDING:  # grounding = deterministic, needs the graph
            try:
                return self._roundtrip(req)
            except Exception as e:  # noqa: BLE001
                log.warning("VLM subprocess failed (%r) -> rule fallback", e)
                self._proc = None
        t0 = time.monotonic()
        ans = self._rule.run(req)
        return VLMResult(req.req_id, req.kind, ans, time.monotonic() - t0, "rule")

    def _roundtrip(self, req: VLMRequest) -> VLMResult:
        deadline = time.monotonic() + self.cfg.request_timeout_s
        if not self._sub_ok:  # wait for model load (first request only)
            msg = self._res_q.get(timeout=max(self.cfg.request_timeout_s, 120.0))
            if msg[0] != "ready":
                raise RuntimeError(msg)
            self._sub_ok = True
        wire = VLMRequest(req.req_id, req.kind,
                          {k: v for k, v in req.payload.items() if k != "graph"},
                          req.created, req.ttl_s, req.key)
        self._req_q.put(wire, timeout=1)
        while time.monotonic() < deadline:
            try:
                msg = self._res_q.get(timeout=0.2)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError("worker died")
                continue
            if msg[0] == "ok" and msg[1].req_id == req.req_id:
                return msg[1]
            if msg[0] == "err":
                raise RuntimeError(msg[2])
        raise TimeoutError("VLM request timed out")
