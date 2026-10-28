# SPDX-License-Identifier: GPL-3.0-only

"""The life-criteria harness: it must report what it cannot measure.

The interesting failure this guards against is a harness that invents a
plausible row for a criterion it cannot run. Homeostasis needs the derived
aggregate state channel, which does not exist yet, so the harness has to say
so and cap the evidence level instead of filling the row with a number.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import life_criteria

from pmc.dignity.protocol import DignityProtocol
from pmc.dignity.records import Instance
from pmc.life_criteria import BatteryReport
from pmc.rhythm import WORK_VERIFY_JOURNAL


def workspace_with(instance_id: str = "inst_001", profile: str = "autonomous") -> Path:
    root = Path(tempfile.mkdtemp(prefix="pmc-life-"))
    protocol = DignityProtocol(root, profile=profile)
    protocol.birth(Instance(instance_id=instance_id, name="Oracle"))
    return root


def rhythm_cycles(root: Path, instance_id: str = "inst_001") -> int:
    return sum(
        1
        for entry in DignityProtocol(root).registry.journal()
        if entry.get("type") == "RHYTHM"
        and entry.get("payload", {}).get("act") == "cycle"
    )


class HarnessTest(unittest.TestCase):
    def test_homeostasis_is_measured_on_the_real_channel(self):
        """The criterion that used to be missing is now measured, and published.

        The series comes from the derived aggregate channel driven through the
        protocol's own API - a quiet start, admissions that raise the declared
        load, digestions that bring it down - so the reading is a real
        excursion rather than an invented one.
        """
        root = workspace_with()
        payload = life_criteria.run(root, "inst_001")
        names = [result["criterion"] for result in payload["results"]]
        self.assertIn("homeostasis", names)
        self.assertNotIn("homeostasis", payload["unmeasured"])
        found = {result["criterion"]: result for result in payload["results"]}
        homeostasis = found["homeostasis"]
        self.assertEqual(homeostasis["detail"]["sample"]["inside"],
                         homeostasis["detail"]["sample"]["total"])
        self.assertLess(homeostasis["null_value"], homeostasis["value"])

    def test_the_band_is_declared_and_narrower_than_the_bounds(self):
        root = workspace_with()
        payload = life_criteria.run(root, "inst_001")
        found = {result["criterion"]: result for result in payload["results"]}
        self.assertLess(found["homeostasis"]["detail"]["bound"], 1.0)

    def test_one_failing_criterion_caps_the_evidence_level(self):
        """A response that never changes anything holds the battery at E1.

        The level is capped by the failing criterion, not by a missing one, so
        the battery can rise when the instance starts doing work that changes
        something.
        """
        root = workspace_with()
        protocol = DignityProtocol(root)
        cycle = protocol.rhythm("inst_001")
        cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
        cycle.tick(force=True)
        payload = life_criteria.run(root, "inst_001")
        self.assertEqual(payload["evidence_level"], "E1")
        failed = [r["criterion"] for r in payload["results"] if not r["passed"]]
        self.assertIn("response", failed)

    def test_the_harness_runs_on_a_real_workspace(self):
        root = workspace_with()
        protocol = DignityProtocol(root)
        cycle = protocol.rhythm("inst_001")
        cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
        cycle.tick(force=True)
        payload = life_criteria.run(root, "inst_001")
        self.assertEqual(payload["profile"], "autonomous")
        self.assertEqual(payload["instance_id"], "inst_001")
        self.assertGreaterEqual(len(payload["results"]), 3)
        for result in payload["results"]:
            self.assertIn("null_value", result)
            self.assertTrue(result["note"])

    def test_continuity_agrees_across_a_restart(self):
        root = workspace_with()
        payload = life_criteria.run(root, "inst_001")
        found = {
            result["criterion"]: result for result in payload["results"]
        }
        continuity = found["continuity"]
        self.assertTrue(continuity["passed"])
        self.assertEqual(continuity["value"], 1.0)
        self.assertLess(continuity["null_value"], 1.0)

    def test_boundaries_pass_on_a_workspace_with_no_foreign_reads(self):
        root = workspace_with()
        payload = life_criteria.run(root, "inst_001")
        found = {
            result["criterion"]: result for result in payload["results"]
        }
        boundaries = found["boundaries"]
        self.assertTrue(boundaries["passed"])
        self.assertEqual(boundaries["detail"]["clean"], boundaries["detail"]["acts"])

    def test_a_dignity_workspace_is_reported_as_such(self):
        root = workspace_with(profile="dignity")
        payload = life_criteria.run(root, "inst_001")
        self.assertEqual(payload["profile"], "dignity")
        # No rhythm means no response observations, so that row is missing too.
        self.assertIn("response", payload["unmeasured"])

    def test_the_report_is_json_serialisable(self):
        root = workspace_with()
        payload = life_criteria.run(root, "inst_001")
        text = json.dumps(payload, indent=2, sort_keys=True)
        self.assertIn("pmc-life-criteria/1", text)

    def test_response_reports_a_negative_when_nothing_changed(self):
        """A rhythm whose cycles change nothing does not respond to the world.

        This is the criterion's honest answer on a fresh workspace, and it is
        the answer that keeps the battery from being a decoration.
        """
        root = workspace_with()
        protocol = DignityProtocol(root)
        cycle = protocol.rhythm("inst_001")
        cycle.enqueue_kind(WORK_VERIFY_JOURNAL)
        cycle.tick(force=True)
        payload = life_criteria.run(root, "inst_001")
        found = {
            result["criterion"]: result for result in payload["results"]
        }
        response = found["response"]
        self.assertFalse(response["passed"])
        self.assertEqual(response["value"], 0.0)

    def test_an_empty_report_has_no_level(self):
        self.assertEqual(BatteryReport(results=()).evidence_level(), "E0")


if __name__ == "__main__":
    unittest.main()
