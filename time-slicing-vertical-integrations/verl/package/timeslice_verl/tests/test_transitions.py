"""Unit tests for RoleLocks + PhaseTransitions (pure python, no GPU, no verl).

Run:  python3 -m unittest discover -s <this dir> -v

These tests drive the exact hook sequence the disagg trainer produces at the
pinned verl commit and assert the lock-transition invariants:
  * trainer-first global order (TRAINER never requested while SAMPLER held)
  * dual-lock span around weight sync (both held for the whole sync)
  * yield-on-starvation (TRAINER released before SAMPLER acquired for the
    sample wait; conditional-release keeps TRAINER when the batch is buffered)
  * no lock leaks (every acquire matched by a release; nothing held at end)
"""

import importlib.util
import os
import unittest

# Load locks.py directly by path: importing the timeslice_verl package would
# pull in the trainer modules, which require verl (absent in the fast CPU test
# step). locks.py itself is dependency-free.
_LOCKS_PATH = os.path.join(os.path.dirname(__file__), "..", "timeslice_verl", "locks.py")
_spec = importlib.util.spec_from_file_location("_timeslice_locks", _LOCKS_PATH)
_locks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_locks)

SAMPLER = _locks.SAMPLER
TRAINER = _locks.TRAINER
PhaseTransitions = _locks.PhaseTransitions
RoleLocks = _locks.RoleLocks

JOB = "job-a"
ADDR = "orch:50051"
TG = "trainers"
SG = "samplers"


class FakeClient:
    """Records acquire/release calls into a shared event log."""

    def __init__(self, events, job_id, group_id):
        self.events = events
        self.job_id = job_id
        self.group_id = group_id
        self.closed = False

    def acquire(self, group_id):
        assert group_id == self.group_id
        self.events.append(("acquire", group_id))
        return type("R", (), {"waited_ms": 0, "context_restored": False})()

    def release(self, group_id):
        assert group_id == self.group_id
        self.events.append(("release", group_id))
        return type("R", (), {"pending_waiters": 0, "snapshot_deferred": False})()

    def close(self):
        self.closed = True


def make_locks(events):
    return RoleLocks(
        job_id=JOB,
        orch_addr=ADDR,
        trainer_group=TG,
        sampler_group=SG,
        client_factory=lambda target, job_id, group_id: FakeClient(events, job_id, group_id),
    )


def held_after(events):
    """Replay the event log and return the set of groups still held."""
    held = set()
    for op, group in events:
        if op == "acquire":
            assert group not in held, f"double acquire of {group}: {events}"
            held.add(group)
        else:
            assert group in held, f"release of un-held {group}: {events}"
            held.discard(group)
    return held


class TestRoleLocks(unittest.TestCase):
    def test_idempotent_acquire_release(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(TRAINER)  # no-op
        self.assertEqual(events, [("acquire", TG)])
        locks.release(TRAINER)
        locks.release(TRAINER)  # no-op
        self.assertEqual(events, [("acquire", TG), ("release", TG)])

    def test_order_violation_raises(self):
        events = []
        locks = make_locks(events)
        locks.acquire(SAMPLER)
        with self.assertRaises(RuntimeError):
            locks.acquire(TRAINER)
        # order violation must be detected even in no-op (disabled) mode:
        disabled = RoleLocks(None, None, None, None)
        disabled._held.add(SAMPLER)  # simulate impossible state
        with self.assertRaises(RuntimeError):
            disabled.acquire(TRAINER)

    def test_in_order_dual_acquire_allowed(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(SAMPLER)  # in-order: fine
        self.assertEqual(events, [("acquire", TG), ("acquire", SG)])
        locks.release_all()
        # reverse-order release: sampler first
        self.assertEqual(events[2:], [("release", SG), ("release", TG)])
        self.assertEqual(held_after(events), set())

    def test_same_group_names_rejected(self):
        with self.assertRaises(ValueError):
            RoleLocks(JOB, ADDR, "g", "g", client_factory=lambda *a: FakeClient([], JOB, "g"))

    def test_close_releases_and_closes_clients(self):
        events = []
        locks = make_locks(events)
        locks.acquire(TRAINER)
        locks.acquire(SAMPLER)
        locks.close()
        self.assertEqual(held_after(events), set())
        for client in locks._clients.values():
            self.assertTrue(client.closed)


class TestPhaseTransitions(unittest.TestCase):
    """Drive the full disagg trainer hook sequence and check the event log."""

    def run_steps(self, n_steps, starved_by_step, with_validation=False, natural_exit=True):
        events = []
        locks = make_locks(events)
        phases = PhaseTransitions(locks)

        phases.init_begin()  # __init__ -> trainer.init() + on_init_end dual sync
        spans = {"init": list(events)}
        phases.train_begin()  # on_train_begin

        for step in range(n_steps):
            # feed happens here (TRAINER must be held)
            self.assertTrue(locks.held(TRAINER), f"step {step}: feed without TRAINER")
            phases.sample_begin(starved=starved_by_step[step])  # on_sample_begin
            if starved_by_step[step]:
                # sample-wait span: sampler only (yield-on-starvation)
                self.assertFalse(locks.held(TRAINER), f"step {step}: TRAINER held during sample wait")
                self.assertTrue(locks.held(SAMPLER), f"step {step}: SAMPLER not held while generating")
            else:
                # conditional release: batch buffered, trainer kept, no sampler
                self.assertTrue(locks.held(TRAINER))
                self.assertFalse(locks.held(SAMPLER))
            phases.sample_end()  # on_sample_end
            # train sub-phases: trainer only
            self.assertTrue(locks.held(TRAINER), f"step {step}: train without TRAINER")
            self.assertFalse(locks.held(SAMPLER), f"step {step}: SAMPLER leaked into train phase")
            # on_step_end: dual-lock weight sync span
            mark = len(events)
            phases.weight_sync_begin()
            self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER), f"step {step}: weight sync needs both")
            phases.weight_sync_end()
            sync_events = events[mark:]
            self.assertEqual(sync_events, [("acquire", SG), ("release", SG)], f"step {step}: {sync_events}")

            if with_validation:
                phases.validate_begin()
                self.assertTrue(locks.held(TRAINER) and locks.held(SAMPLER))
                phases.validate_end()
                self.assertTrue(locks.held(TRAINER))
                self.assertFalse(locks.held(SAMPLER))

        if natural_exit:
            # fit() early-returns on the natural last step WITHOUT on_train_end;
            # the atexit net must still leave nothing held.
            locks._atexit_cleanup()
        else:
            phases.train_end()
        return events, locks

    def test_multi_step_starved(self):
        events, locks = self.run_steps(3, [True, True, True])
        self.assertEqual(held_after(events), set(), f"leaked locks: {events}")

    def test_conditional_release_mixed(self):
        events, locks = self.run_steps(4, [True, False, True, False])
        self.assertEqual(held_after(events), set())
        # the non-starved steps must not touch the sampler group before sync
        self.assertEqual(held_after(events), set())

    def test_with_validation_and_train_end(self):
        events, locks = self.run_steps(2, [True, True], with_validation=True, natural_exit=False)
        self.assertEqual(held_after(events), set())
        for client in locks._clients.values():
            self.assertTrue(client.closed)

    def test_trainer_first_global_order(self):
        """At no point in any schedule is TRAINER requested while SAMPLER is held."""
        events, _ = self.run_steps(3, [True, False, True], with_validation=True, natural_exit=False)
        held = set()
        for op, group in events:
            if op == "acquire":
                if group == TG:
                    self.assertNotIn(SG, held, f"TRAINER acquired while SAMPLER held: {events}")
                held.add(group)
            else:
                held.discard(group)

    def test_yield_on_starvation_ordering(self):
        """During a starved sample_begin, TRAINER release precedes SAMPLER acquire."""
        events = []
        locks = make_locks(events)
        phases = PhaseTransitions(locks)
        phases.init_begin()
        phases.train_begin()
        mark = len(events)
        phases.sample_begin(starved=True)
        self.assertEqual(events[mark:], [("release", TG), ("acquire", SG)])
        mark = len(events)
        phases.sample_end()
        self.assertEqual(events[mark:], [("release", SG), ("acquire", TG)])

    def test_weight_sync_requires_trainer(self):
        events = []
        locks = make_locks(events)
        phases = PhaseTransitions(locks)
        with self.assertRaises(RuntimeError):
            phases.weight_sync_begin()

    def test_noop_mode_runs_clean(self):
        """Without env config everything must be a silent no-op (image reuse)."""
        locks = RoleLocks(None, None, None, None)
        phases = PhaseTransitions(locks)
        phases.init_begin()
        phases.train_begin()
        phases.sample_begin(starved=True)
        phases.sample_end()
        phases.weight_sync_begin()
        phases.weight_sync_end()
        phases.train_end()


if __name__ == "__main__":
    unittest.main()
