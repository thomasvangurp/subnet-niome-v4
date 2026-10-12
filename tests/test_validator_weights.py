from __future__ import annotations

import asyncio
import importlib
import unittest
from unittest import mock

from niome_subnet.genomics.model import MinerScore
from niome_subnet.genomics.validation.errors import ValidatorFault

# `niome_subnet.validator.forward` is re-exported in the package __init__ as the
# `forward` function, so import the module explicitly rather than the attribute.
forward_mod = importlib.import_module("niome_subnet.validator.forward")


class _FakeValidator:
    """Minimal stand-in for the validator neuron `run_validation` operates on."""

    def __init__(self, miner_uids, task_id="task-1"):
        self.miner_uids = miner_uids
        self.task_id = task_id
        self.set_weights_calls = []

    def save_state(self):
        pass

    def set_weights(self, scores, task_id):
        self.set_weights_calls.append((scores, task_id))


def _score(uid):
    return MinerScore(uid=uid, breakdown={}, final_score=1.0 - 0.1 * uid, log="")


class RunValidationWeightsTests(unittest.TestCase):
    """`run_validation` must never publish an owner-only weight vector.

    The regression this guards against: every miner failing to score (or scoring
    zero) used to leave `scores` empty/all-zero, which `set_weights` then turned
    into a 100%-owner vector on chain. Scoring failures must instead abandon the
    round and keep the previously-committed weights.
    """

    def _run(self, fake, scorer):
        with (
            mock.patch.object(forward_mod, "boto3") as boto3,
            mock.patch.object(forward_mod, "benchmark_submission", side_effect=scorer),
            mock.patch.object(
                forward_mod, "upload_final_submissions_to_server"
            ) as upload,
        ):
            boto3.client.return_value.download_file = mock.Mock()
            asyncio.run(forward_mod.run_validation(fake))
        return upload

    def test_scoring_failures_do_not_set_weights(self):
        fake = _FakeValidator([0, 1, 2])

        def scorer(uid):
            raise ValueError(f"boom for {uid}")

        self._run(fake, scorer)
        self.assertEqual([], fake.set_weights_calls)

    def test_all_zero_scores_do_not_set_weights(self):
        fake = _FakeValidator([0, 1])

        def scorer(uid):
            return MinerScore(uid=uid, breakdown={}, final_score=0.0, log="")

        self._run(fake, scorer)
        self.assertEqual([], fake.set_weights_calls)

    def test_validator_fault_abandons_without_setting_weights(self):
        fake = _FakeValidator([0, 1])

        def scorer(uid):
            raise ValidatorFault("broken validator environment")

        self._run(fake, scorer)
        self.assertEqual([], fake.set_weights_calls)

    def test_positive_scores_set_weights(self):
        fake = _FakeValidator([0, 1])
        upload = self._run(fake, _score)

        self.assertEqual(1, len(fake.set_weights_calls))
        scores, task_id = fake.set_weights_calls[0]
        self.assertEqual("task-1", task_id)
        self.assertEqual([0, 1], [s.uid for s in scores])
        upload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
