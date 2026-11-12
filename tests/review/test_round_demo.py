import json
import unittest

from examples.round_demo import run


class RoundDemoTests(unittest.TestCase):
    def test_demo_round_is_reproducible_and_informative(self):
        first = run()
        self.assertTrue(first['synthetic'])
        self.assertEqual(first['pinned']['run_id'], 'september-run')
        self.assertGreater(first['training']['documents'], 0)
        self.assertEqual(first['backtest']['verdict']['worth_enabling'], False)
        self.assertEqual(
            {reason['code'] for reason in first['backtest']['verdict']['reasons']},
            {'no_precision_gain_over_isolated_baselines', 'no_recall_gain_over_isolated_baselines'},
        )
        self.assertLessEqual(first['selection']['used_minutes'], first['selection']['budget_minutes'])
        self.assertTrue(first['selection']['one_per_cluster'])
        self.assertTrue(first['selection']['source_floor']['satisfied'])
        self.assertEqual(len(first['selection']['chosen']), len({item['cluster_id'] for item in first['selection']['chosen']}))
        self.assertTrue(first['creation_mutates_no_document']['unchanged'])
        self.assertEqual(first['conflict']['code'], 'round_source_floor_infeasible')
        self.assertEqual(first['lease_review']['state'], 'accepted')
        reasons = {reason['code'] for item in first['invalidation_after_new_version_and_withdrawal'] for reason in item['reasons']}
        self.assertIn('candidate_version_superseded', reasons)
        self.assertIn('document_withdrawn', reasons)
        self.assertTrue(first['repeat_create']['same_digest'])
        self.assertTrue(first['evidence']['stored_evidence_valid'])
        self.assertTrue(first['evidence']['selection_recheck'])

    def test_demo_output_is_stable_json(self):
        first = run()
        second = run()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
