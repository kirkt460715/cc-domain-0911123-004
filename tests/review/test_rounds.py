import unittest

from tests.support import Case, record
from corpus_gate import documents, batch
from corpus_gate.review import desk, risk, round as rounds
from corpus_gate.dedup import clusters


SHORT = 'short'
CONTACT = 'contact synthetic@example.com for alpha beta gamma help'
CLEAN = 'calibration procedure stable sensor reference text alpha'


class RoundTests(Case):
    def prepare(self, values, decided, batch_id='batch', source='source'):
        self.ingest([record('d' + str(index), value, minutes=index % 4 + 1) for index, value in enumerate(values)], batch_id, source)
        desk.register(self.db, 'reviewer', 'supervisor')
        for index, decision in decided.items():
            self.time = f'2026-11-11T00:00:{index + 10:02d}.000000Z'
            document = self.document('d' + str(index))
            token = desk.claim(self.db, document['document_id'], 'reviewer', 1)
            codes = sorted({finding['code'] for finding in document['findings']})
            desk.decide(self.db, document['document_id'], token, document['revision'], decision, 'Synthetic decision', codes)
        self.time = '2026-11-12T00:00:00.000000Z'

    def test_round_trains_only_on_cutoff_human_decisions(self):
        self.prepare([SHORT, CONTACT, CLEAN, 'other distinct words for a fourth record'], {0: 'accept', 1: 'reject'})
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:02:00.000000Z', budget_minutes=10, seed='seed')
        self.assertEqual(evidence['training']['documents'], 2)
        self.assertEqual(evidence['training']['labels'], {'reject': 1, 'accept': 1})
        before = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=10, seed='seed')
        self.assertEqual(before['training']['documents'], 0)
        self.assertEqual(before['training']['excluded_decisions'][0]['reason'], 'after_cutoff')
        self.assertEqual(before['model']['status'], 'constant_single_class')
        self.assertIn('single_class_labels', [item['code'] for item in before['model']['fallbacks']])
        self.assertEqual(before['backtest']['verdict']['worth_enabling'], False)

    def test_rule_findings_never_become_training_truth(self):
        self.ingest([record('a', SHORT), record('b', CONTACT), record('c', CLEAN)])
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=10, seed='seed')
        self.assertEqual(evidence['training']['documents'], 0)
        self.assertEqual(evidence['model']['training_documents'], 0)
        self.assertGreaterEqual(len(evidence['candidates']), 2)
        quarantined = [row for row in evidence['candidates'] if row['document_id'] == self.document('a')['document_id']][0]
        self.assertEqual(quarantined['rule']['quarantine_findings'], 1)
        self.assertEqual(quarantined['prediction']['fallback'], 'constant_single_class')

    def test_withdrawal_and_superseded_versions_are_not_reused(self):
        self.prepare([SHORT, CONTACT], {0: 'reject', 1: 'accept'})
        first = self.document('d0')['document_id']
        documents.withdraw(self.db, first, documents.get(self.db, first)['revision'], 'Retracted label')
        self.ingest([record('d1', CONTACT, version=2, minutes=2)], 'new-version')
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-12T00:00:00.000000Z', budget_minutes=10, seed='seed')
        self.assertEqual(evidence['training']['documents'], 0)
        reasons = {row['reason'] for row in evidence['training']['excluded_decisions']}
        self.assertIn('document_withdrawn', reasons)
        self.assertIn('superseded_version', reasons)

    def test_unknown_and_empty_text_have_explanations(self):
        self.prepare([SHORT, CONTACT, CLEAN], {0: 'reject', 1: 'accept'})
        self.ingest([record('unknown', 'zzzz qqqq'), record('empty', '')], 'edge')
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-12T00:00:00.000000Z', budget_minutes=10, seed='seed')
        self.assertEqual(evidence['model']['status'], 'fitted')
        rows = {row['document_id']: row for row in evidence['candidates']}
        unknown = rows[self.document('unknown')['document_id']]['prediction']
        empty = rows[self.document('empty')['document_id']]['prediction']
        self.assertEqual(unknown['fallback'], 'unknown_vocabulary')
        self.assertEqual(unknown['known_terms'], 0)
        self.assertIn('outside the training vocabulary', unknown['explanation'])
        self.assertEqual(empty['fallback'], 'empty_text')
        self.assertEqual(empty['tokens'], 0)
        self.assertTrue(0.0 <= unknown['probability'] <= 1.0)
        self.assertTrue(0.0 <= empty['probability'] <= 1.0)

    def test_budget_cluster_and_source_floor_conflicts(self):
        values = [SHORT, CONTACT, CLEAN, 'distinct words make another document here', 'a fifth different text sample goes here']
        self.ingest([record('d' + str(index), value, minutes=5) for index, value in enumerate(values)])
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=10, seed='seed', exploration_fraction=0.2, source_floor=['source'], floor_documents=1)
        self.assertLessEqual(evidence['selection']['used_minutes'], 10)
        self.assertLessEqual(len(evidence['selection']['chosen']), 2)
        self.assertEqual(len({item['cluster_id'] for item in evidence['selection']['chosen']}), len(evidence['selection']['chosen']))
        self.assertTrue(evidence['selection']['constraints']['one_per_cluster'])
        self.assertTrue(evidence['selection']['source_floor']['satisfied'])
        self.fails('round_source_floor_infeasible', rounds.plan, self.db, round_id='tight', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=4, seed='seed', source_floor=['source'], floor_documents=1)

    def test_floor_conflict_when_source_has_no_candidates(self):
        self.ingest()
        clusters.run(self.db, 'run')
        self.fails('round_source_floor_infeasible', rounds.plan, self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=10, seed='seed', source_floor=['missing'])

    def test_create_is_idempotent_and_mutates_no_documents(self):
        self.ingest([record('a', SHORT), record('b', CONTACT)])
        clusters.run(self.db, 'run')
        request = dict(round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=6, seed='seed', exploration_fraction=0.2)
        states = {row['document_id']: row['state'] for row in self.db.all('SELECT document_id,state FROM documents')}
        first = rounds.create(self.db, **request)
        second = rounds.create(self.db, **request)
        self.assertEqual(first['digest'], second['digest'])
        self.assertEqual(self.db.scalar('SELECT COUNT(*) FROM review_round_queue WHERE round_id=?', ('round',)), len(first['queue']))
        self.assertEqual(states, {row['document_id']: row['state'] for row in self.db.all('SELECT document_id,state FROM documents')})
        self.fails('round_conflict', rounds.create, self.db, **{**request, 'budget_minutes': 7})
        evidence = rounds.evidence(self.db, 'round')
        self.assertTrue(evidence['stored_evidence_valid'])
        self.assertTrue(evidence['selection_recheck']['matches_stored_queue'])
        self.assertTrue(evidence['selection_recheck']['within_budget'])
        self.assertTrue(evidence['selection_recheck']['one_per_cluster'])
        self.assertTrue(evidence['replay_now']['matches_stored'])
        self.assertEqual(db_events := [event['action'] for event in first['events']], ['created'])

    def test_validate_reports_version_claim_and_withdrawal(self):
        self.ingest([record('a', SHORT, minutes=1), record('b', CONTACT, minutes=1), record('c', CLEAN, minutes=1), record('d', 'another distinct text value here', minutes=1)])
        clusters.run(self.db, 'run')
        created = rounds.create(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=5, seed='seed', exploration_fraction=0.0)
        self.assertTrue(rounds.validate(self.db, 'round')['valid'])
        desk.register(self.db, 'reviewer', 'supervisor')
        claimed = created['queue'][0]['document_id']
        desk.claim(self.db, claimed, 'reviewer', 1)
        report = rounds.validate(self.db, 'round')
        claimed_reasons = next(item for item in report['invalid'] if item['document_id'] == claimed)['reasons']
        self.assertIn('currently_claimed', [reason['code'] for reason in claimed_reasons])
        withdrawn = next(item['document_id'] for item in created['queue'] if item['document_id'] != claimed)
        documents.withdraw(self.db, withdrawn, documents.get(self.db, withdrawn)['revision'], 'Retracted')
        report = rounds.validate(self.db, 'round')
        withdrawn_reasons = next(item for item in report['invalid'] if item['document_id'] == withdrawn)['reasons']
        self.assertIn('document_withdrawn', [reason['code'] for reason in withdrawn_reasons])

    def test_complete_round_uses_lease_and_does_not_auto_decide(self):
        self.ingest([record('q0', 'alpha', minutes=2), record('q1', 'beta', minutes=2), record('q2', 'gamma', minutes=3), record('q3', 'delta', minutes=3)])
        clusters.run(self.db, 'run')
        desk.register(self.db, 'reviewer', 'supervisor')
        created = rounds.create(self.db, round_id='round', run_id='run', cutoff='2026-11-12T00:00:00.000000Z', budget_minutes=8, seed='seed', exploration_fraction=0.25, source_floor=['source'], floor_documents=1)
        self.assertTrue(created['queue'])
        for item in created['queue']:
            self.assertIn(documents.get(self.db, item['document_id'])['state'], {'quarantined', 'rejected'})
        first = created['queue'][0]['document_id']
        token = desk.claim(self.db, first, 'reviewer', 1)
        document = documents.get(self.db, first)
        result = desk.decide(self.db, first, token, document['revision'], 'reject', 'Lease decision', sorted({finding['code'] for finding in document['findings']}))
        self.assertEqual(result['state'], 'rejected')
        self.assertEqual(self.db.scalar('SELECT COUNT(*) FROM review_events WHERE document_id=?', (first,)), 1)

    def test_model_is_enabled_only_when_it_beats_cluster_isolated_baselines(self):
        texts = []
        decisions = {}
        for index in range(48):
            bad = index % 2 == 0
            text = ('leak warning sensor ' if bad else 'calibration stable sensor ') + 'sample ' + ('term' + str(index // 4)) * 2
            texts.append(text)
            decisions[index] = 'reject' if bad else 'accept'
        # Three decision days give the forward check two nonempty folds.
        from corpus_gate.text import profiles
        profiles.put(self.db, 'wide', 0, {'quality': {'minimum_tokens': 100}})
        self.ingest([record('m' + str(index), text, minutes=index % 4 + 1) for index, text in enumerate(texts)], profile='wide')
        desk.register(self.db, 'reviewer', 'supervisor')
        for index, decision in sorted(decisions.items()):
            day = {0: '2026-11-11T00:00:10.000000Z', 1: '2026-11-11T00:00:20.000000Z', 2: '2026-11-11T00:00:30.000000Z'}[index % 3]
            self.time = day
            document = self.document('m' + str(index))
            token = desk.claim(self.db, document['document_id'], 'reviewer', 1)
            desk.decide(self.db, document['document_id'], token, document['revision'], decision, 'Synthetic decision', sorted({finding['code'] for finding in document['findings']}))
        self.time = '2026-11-12T00:00:00.000000Z'
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:30.000000Z', budget_minutes=20, seed='seed', default_minutes=3)
        self.assertEqual(evidence['strategy']['name'], 'text_model_active')
        self.assertTrue(evidence['strategy']['model_enabled'])
        self.assertEqual(evidence['strategy']['decision']['reasons'], [])
        model = evidence['backtest']['aggregate']['strategies']['model_isolated']
        rule = evidence['backtest']['aggregate']['strategies']['rule_isolated']
        self.assertGreater(model['precision'], rule['precision'])
        self.assertGreater(model['recall'], rule['recall'])
        for item in evidence['selection']['chosen']:
            self.assertIn(item['reasons'][0]['code'], {'source_floor', 'fixed_seed_exploration', 'ranking_score'})

    def test_operations_are_registered(self):
        operations = self.app.operations()
        for name in ('review.round_plan', 'review.round_create', 'review.round_get', 'review.round_evidence', 'review.round_validate', 'review.round_close'):
            self.assertIn(name, operations)

    def test_evidence_is_digest_bound(self):
        self.ingest([record('a', SHORT), record('b', CLEAN)])
        clusters.run(self.db, 'run')
        rounds.create(self.db, round_id='round', run_id='run', cutoff='2026-11-11T00:00:00.000000Z', budget_minutes=6, seed='seed')
        self.db.execute("UPDATE review_rounds SET evidence='{}' WHERE round_id='round'")
        self.fails('round_corrupt', rounds.get, self.db, 'round')

    def test_single_class_model_fallback_ranking_is_explainable(self):
        self.prepare([SHORT, CONTACT], {0: 'reject', 1: 'reject'})
        clusters.run(self.db, 'run')
        evidence = rounds.plan(self.db, round_id='round', run_id='run', cutoff='2026-11-12T00:00:00.000000Z', budget_minutes=6, seed='seed')
        self.assertEqual(evidence['model']['status'], 'constant_single_class')
        for row in evidence['candidates']:
            self.assertEqual(row['prediction']['fallback'], 'constant_single_class')
            self.assertGreaterEqual(row['prediction']['probability'], 0.5)
        self.assertEqual(evidence['backtest']['verdict']['worth_enabling'], False)
        self.assertEqual(evidence['strategy']['name'], 'rule_priority_isolated')


if __name__ == '__main__':
    unittest.main()
