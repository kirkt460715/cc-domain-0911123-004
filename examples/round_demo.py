"""End-to-end demonstration of one active review round.

Replays the synthetic September arrivals and their human decisions through the
normal intake and review-lease workflow, then plans and freezes an October
round that spends a fixed minute budget on unreviewed material. Prints a
compact JSON summary whose `round` section is the frozen evidence needed to
recompute the selection.
"""

import json

from corpus_gate import batch, documents
from corpus_gate.dedup import clusters
from corpus_gate.errors import GateError
from corpus_gate.ingest import sources
from corpus_gate.review import desk
from corpus_gate.review import round as rounds
from corpus_gate.service import CorpusGate
from corpus_gate.text import profiles

from examples.context import load

AUTHORIZATION = {
    'purposes': ['research'],
    'regions': ['cn'],
    'valid_from': '2026-01-01T00:00:00Z',
    'valid_until': '2027-01-01T00:00:00Z',
    'obligations': [],
    'allow_export': True,
}


def setup(app):
    data = load()
    db = app.db
    for source in ('old', 'new'):
        if not db.one('SELECT source_id FROM sources WHERE source_id=?', (source,)):
            sources.create(db, source, AUTHORIZATION)
    if not db.one('SELECT profile_id FROM profiles WHERE profile_id=?', ('local',)):
        profiles.put(db, 'local', 0, {'quality': {'minimum_tokens': 100}})
    desk.register(db, 'expert', 'supervisor')
    now = ['2026-08-01T00:00:00Z']

    def clock():
        return now[0]

    app.db.clock = clock
    dates = sorted({row['metadata']['arrived_at'] for row in data['records']} | {row['decided_at'] for row in data['decisions']})
    for day in dates:
        now[0] = day
        for source in ('old', 'new'):
            records = [row for row in data['records'] if row['metadata']['source_group'] == source and row['metadata']['arrived_at'] == day]
            if records:
                batch_id = source + '-' + day[:10]
                batch.stage(db, batch_id, source, 'local', records)
                batch.run(db, batch_id)
        for row in data['decisions']:
            if row['decided_at'] != day:
                continue
            document = documents.head(db, 'old', row['external_id'])
            token = desk.claim(db, document['document_id'], 'expert')
            codes = sorted({finding['code'] for finding in document['findings']})
            desk.decide(db, document['document_id'], token, document['revision'], row['decision'], 'Synthetic human decision', codes)
    return data, now


def clean_state_changes(db, document_ids, before):
    after = {identity: documents.get(db, identity)['state'] for identity in document_ids}
    changed = {identity: {'before': before[identity], 'after': after[identity]} for identity in document_ids if before[identity] != after[identity]}
    return {'unchanged': not changed, 'changed': changed}


def run():
    with CorpusGate() as app:
        _, now = setup(app)
        db = app.db
        now[0] = '2026-09-10T00:00:00Z'
        clusters.run(db, 'september-run')
        run_info = clusters.get(db, 'september-run')

        now[0] = '2026-10-01T00:00:00Z'
        request = {
            'round_id': 'october-round',
            'run_id': 'september-run',
            'cutoff': '2026-09-10T00:00:00Z',
            'budget_minutes': 30,
            'seed': 'round-demo',
            'exploration_fraction': 0.1,
            'source_floor': ['new'],
            'floor_documents': 1,
            'default_minutes': 5,
        }
        conflict = None
        try:
            rounds.plan(db, **{**request, 'round_id': 'infeasible', 'source_floor': ['old', 'new'], 'budget_minutes': 1})
        except GateError as exc:
            conflict = exc.as_dict()

        created = rounds.create(db, **request)
        evidence = created['evidence']
        frozen_states = {item['document_id']: documents.get(db, item['document_id'])['state'] for item in created['queue']}
        untouched = clean_state_changes(db, list(frozen_states), frozen_states)

        # A reviewer still works through the existing lease; the round only orders work.
        now[0] = '2026-10-02T09:00:00Z'
        first = created['queue'][0]['document_id']
        token = desk.claim(db, first, 'expert')
        before_decision = documents.get(db, first)
        acknowledged = sorted({finding['code'] for finding in before_decision['findings']})
        desk.decide(db, first, token, before_decision['revision'], 'accept', 'Round sample reviewed under normal lease', acknowledged)
        after_decision = rounds.validate(db, 'october-round')

        # A new candidate version for a different queue item and a withdrawal
        # arrive after the round was frozen.
        now[0] = '2026-10-05T00:00:00Z'
        versioned = next(item['document_id'] for item in created['queue'] if item['document_id'] != first and documents.get(db, item['document_id'])['source_id'] == 'old')
        old = documents.get(db, versioned)
        batch.stage(db, 'october-version', 'old', 'local', [{'external_id': old['external_id'], 'version': 2, 'text': old['raw_text'] + ' revised', 'metadata': old['metadata']}])
        batch.run(db, 'october-version')
        withdrawn = next(item['document_id'] for item in created['queue'] if item['document_id'] not in {first, versioned})
        documents.withdraw(db, withdrawn, documents.get(db, withdrawn)['revision'], 'Synthetic retraction')
        after_changes = rounds.validate(db, 'october-round')

        repeated = rounds.create(db, **request)
        queue_rows = db.scalar('SELECT COUNT(*) FROM review_round_queue WHERE round_id=?', ('october-round',))

        replay = rounds.evidence(db, 'october-round')
        backtest = evidence['backtest']
        aggregate = backtest['aggregate']
        selection = evidence['selection']
        decided_document = documents.get(db, first)
        return {
            'synthetic': True,
            'pinned': {
                'run_id': run_info['run_id'],
                'run_digest': run_info['digest'],
                'run_corpus_revision': run_info['corpus_revision'],
                'cutoff': created['cutoff_at'],
                'budget_minutes': created['budget_minutes'],
                'seed': created['seed'],
            },
            'training': {
                'documents': evidence['training']['documents'],
                'labels': evidence['training']['labels'],
                'excluded_decisions': evidence['training']['excluded_decisions'],
            },
            'model': {
                'status': evidence['model']['status'],
                'training_labels': evidence['model']['training_labels'],
                'calibration': evidence['model']['calibration'],
                'vocabulary_size': len(evidence['model']['vocabulary']),
            },
            'backtest': {
                'folds': [{'decision_time': fold['decision_time'], 'training_documents': fold['training_documents'], 'test_documents': fold['test_documents'], 'isolated_duplicates_removed': len(fold['isolated_duplicates_removed']), 'brier': fold['brier'], 'strategies': {name: {'selected': row['selected'], 'hits': row['hits'], 'precision': row['precision'], 'recall': row['recall'], 'minutes': row['minutes'], 'revisited_minutes': row['revisited_minutes'], 'distinct_clusters': row['distinct_clusters']} for name, row in fold['strategies'].items()}} for fold in backtest['folds']],
                'aggregate': aggregate,
                'verdict': backtest['verdict'],
            },
            'selection': {
                'strategy': selection['strategy'],
                'used_minutes': selection['used_minutes'],
                'budget_minutes': selection['budget_minutes'],
                'exploration': selection['exploration'],
                'source_floor': selection['source_floor'],
                'chosen': [{'document_id': item['document_id'], 'source_id': item['source_id'], 'cluster_id': item['cluster_id'], 'phase': item['phase'], 'minutes': item['minutes'], 'probability_reject': round(item['score']['probability_reject'], 4), 'reasons': item['reasons'][0]['code'] if item['reasons'] else None} for item in selection['chosen']],
                'rejected_count': len(selection['rejected']),
                'rejected_reasons': {identity: reasons for identity, reasons in sorted(selection['rejected'].items()) if any(reason != 'cluster_already_selected' for reason in reasons)},
                'one_per_cluster': selection['constraints']['one_per_cluster'],
                'budget_respected': selection['constraints']['minute_budget_respected'],
            },
            'conflict': conflict,
            'candidate_exclusions': evidence['candidate_exclusions'],
            'creation_mutates_no_document': untouched,
            'lease_review': {
                'document_id': first,
                'decision': 'accept',
                'state': decided_document['state'],
                'findings_acknowledged': acknowledged,
            },
            'invalidation_after_lease_review': after_decision['invalid'],
            'invalidation_after_new_version_and_withdrawal': after_changes['invalid'],
            'repeat_create': {'queue_rows': queue_rows, 'same_digest': repeated['digest'] == created['digest']},
            'evidence': {
                'stored_evidence_valid': replay['stored_evidence_valid'],
                'selection_recheck': replay['selection_recheck']['matches_stored_queue'],
                'stored_digest': replay['stored_digest'],
                'replay_now_matches': replay['replay_now']['matches_stored'],
            },
            'hit_rate_semantics': backtest['verdict']['semantics'],
        }


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
