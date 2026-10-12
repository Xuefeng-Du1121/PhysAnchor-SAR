import unittest

import numpy as np

from peva_sim.audit_simulated_collection import verify_episode_arrays


class VariableCommunicationAuditTests(unittest.TestCase):
    def make_case(self, transmissions):
        config = {
            "dt_s": 1.0, "horizon_s": 2.0, "n_targets": 1,
            "n_uavs": 3, "payload_budget_bytes": 64,
            "header_bytes": 16, "grid_side": 2,
            "survival_tau_s": 10.0,
        }
        shared_uplink = 2 * 3 * (16 + 16)
        shared_downlink = 2 * (2 ** 2 * 4 + 8 + 16)
        record = {
            "steps": 2, "return_value": 0.0,
            "info": {
                "time_s": 2.0, "success_rate": 0.0,
                "ttd_all": 1.0, "survival_score": 0.0,
                "bytes": {
                    "peer_payload": transmissions * 42,
                    "peer_header": transmissions * 16,
                    "shared_uplink": shared_uplink,
                    "shared_downlink": shared_downlink,
                },
            },
        }
        return config, record

    def test_actual_gate_count_drives_payload_and_header(self):
        config, record = self.make_case(transmissions=2)
        verify_episode_arrays(
            np.asarray([0., 1.]),
            np.zeros((2, 1), dtype=np.bool_),
            record, config, peer_payload_bytes=42,
            peer_transmissions=2)

    def test_transmission_count_cannot_exceed_agent_steps(self):
        config, record = self.make_case(transmissions=7)
        with self.assertRaisesRegex(ValueError, "transmission count"):
            verify_episode_arrays(
                np.asarray([0., 1.]),
                np.zeros((2, 1), dtype=np.bool_),
                record, config, peer_payload_bytes=42,
                peer_transmissions=7)


if __name__ == "__main__":
    unittest.main()
