"""Active review rounds: reproducible, budgeted, leakage-aware sampling.

A round pins a candidate document version, a dedup run, a human-review cutoff
and a minute budget. It never accepts or rejects documents; reviewers still
claim and decide through the existing lease workflow.
"""

import json
import math

from corpus_gate import documents
from corpus_gate.dedup import clusters, constraints, similarity
from corpus_gate.errors import GateError, canonical, digest, fail, identifier, instant, integer, number, shape
from corpus_gate.review import risk

MODEL = {'l2': 0.05, 'learning_rate': 0.5, 'epochs': 200, 'min_document_frequency': 1, 'max_vocabulary': 500}
THRESHOLDS = {'minimum_eval_documents': 8, 'minimum_brier_gain': 0.001}
REVIEWABLE = {'quarantined', 'rejected'}
DECISION_STATE = {'accept': 'accepted', 'reject': 'rejected'}


def _mapping(value, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        fail('round_settings', 'Expected settings object', field=field)
    return value


def settings(value):
    shape(value, {'round_id', 'run_id', 'cutoff', 'budget_minutes', 'seed'},
          {'exploration_fraction', 'source_floor', 'floor_documents', 'model', 'thresholds', 'default_minutes', 'coverage_weight'})
    round_id = identifier(value['round_id'])
    run_id = identifier(value['run_id'], 'run_id')
    cutoff = instant(value['cutoff'])
    budget = integer(value['budget_minutes'], 'budget_minutes', 1, 100000)
    seed = identifier(value['seed'], 'seed')
    exploration = number(value.get('exploration_fraction', 0.1), 'exploration_fraction')
    floors = value.get('source_floor', [])
    if not isinstance(floors, list) or len(floors) > 1000:
        fail('round_source_floor', 'source_floor must be a bounded identifier list')
    floors = sorted({identifier(item, 'source_floor') for item in floors})
    floor_documents = integer(value.get('floor_documents', 1), 'floor_documents', 1, 1000)
    model = {**MODEL, **_mapping(value.get('model'), 'model')}
    shape(model, set(MODEL))
    integer(model['epochs'], 'epochs', 1, 2000)
    integer(model['min_document_frequency'], 'min_document_frequency', 1, 100)
    integer(model['max_vocabulary'], 'max_vocabulary', 1, 100000)
    model['l2'] = number(model['l2'], 'l2', 0.0, 10.0)
    model['learning_rate'] = number(model['learning_rate'], 'learning_rate', 0.000001, 10.0)
    thresholds = {**THRESHOLDS, **_mapping(value.get('thresholds'), 'thresholds')}
    shape(thresholds, set(THRESHOLDS))
    integer(thresholds['minimum_eval_documents'], 'minimum_eval_documents', 1, 100000)
    thresholds['minimum_brier_gain'] = number(thresholds['minimum_brier_gain'], 'minimum_brier_gain', -1.0, 1.0)
    default_minutes = integer(value.get('default_minutes', 10), 'default_minutes', 1, 100000)
    coverage_weight = number(value.get('coverage_weight', 0.5), 'coverage_weight', 0.0, 10.0)
    return {
        'round_id': round_id,
        'run_id': run_id,
        'cutoff': cutoff,
        'budget_minutes': budget,
        'seed': seed,
        'exploration_fraction': exploration,
        'source_floor': floors,
        'floor_documents': floor_documents,
        'model': model,
        'thresholds': thresholds,
        'default_minutes': default_minutes,
        'coverage_weight': coverage_weight,
    }


def _request_digest(options):
    return digest({'format': 'review-round-request', 'version': 1, 'options': options})


def _active_claim(db, document_id):
    row = db.one('SELECT * FROM review_claims WHERE document_id=?', (document_id,))
    if row and instant(row['expires_at']) > instant(db.clock()):
        return row
    return None


def _label_candidates(db, cutoff):
    """Human decisions usable as training truth, with auditable exclusions.

    A decision is reusable only when it is final for the exact document
    version: the document is the current head, its revision equals the event
    revision, the decision agrees with the current state, and the version is
    not withdrawn. Rule findings are never labels.
    """
    cutoff = instant(cutoff)
    usable = []
    excluded = []
    for event in db.all('SELECT * FROM review_events ORDER BY event_id'):
        document = documents.get(db, event['document_id'])
        decided_at = instant(event['at'])
        raw_minutes = document['metadata'].get('minutes')
        minutes = raw_minutes if type(raw_minutes) is int and 1 <= raw_minutes <= 100000 else None
        reason = None
        if event['decision'] not in DECISION_STATE:
            reason = 'not_human_accept_or_reject'
        elif decided_at > cutoff:
            reason = 'after_cutoff'
        elif document['state'] == 'withdrawn':
            reason = 'document_withdrawn'
        elif not document['is_head']:
            reason = 'superseded_version'
        elif document['revision'] != event['revision']:
            reason = 'not_final_revision'
        elif document['state'] != DECISION_STATE[event['decision']]:
            reason = 'decision_superseded'
        if reason:
            excluded.append({'document_id': document['document_id'], 'event_id': event['event_id'], 'decision': event['decision'], 'decided_at': decided_at, 'reason': reason})
            continue
        summary = risk.rule_summary(document['findings'])
        usable.append({
            'document_id': document['document_id'],
            'event_id': event['event_id'],
            'text': document['normalized'],
            'label': 1 if event['decision'] == 'reject' else 0,
            'decision': event['decision'],
            'decided_at': decided_at,
            'minutes': minutes,
            'minutes_source': 'metadata' if minutes is not None else 'not_applicable',
            'created_at': document['created_at'],
            'source_id': document['source_id'],
            'rule': summary,
            'stratum': risk.rule_stratum(summary),
        })
    usable.sort(key=lambda row: (row['decided_at'], row['document_id']))
    return usable, excluded


def _partition(db, run, universe, cutoff):
    """Near-duplicate partition pinned to the round's dedup run.

    Documents present in the run keep their run cluster. Documents outside it
    (typically those that were not accepted heads when the run was built) are
    linked by exact hash and by the run's own shingle threshold, so training
    and test isolation still hold. Manual `different` constraints are
    respected, and any suppression is recorded rather than silently ignored.
    """
    run_members = {member['document_id']: member['cluster_id'] for member in run['members']}
    sets = clusters.DisjointSets(universe)
    for cluster_id in sorted(set(run_members[identity] for identity in universe if identity in run_members)):
        identities = sorted(identity for identity in universe if run_members.get(identity) == cluster_id)
        for identity in identities[1:]:
            sets.union(identities[0], identity)
    rules = constraints.active(db, set(universe))
    forbidden = [(rule['left_id'], rule['right_id']) for rule in rules if rule['kind'] == 'different']
    suppressed = []
    for rule in rules:
        if rule['kind'] == 'same':
            if constraints.can_union(sets, rule['left_id'], rule['right_id'], forbidden):
                sets.union(rule['left_id'], rule['right_id'])
            else:
                suppressed.append({'left_id': rule['left_id'], 'right_id': rule['right_id'], 'reason': 'constraint_contradiction'})
    records = {identity: documents.get(db, identity) for identity in universe}
    exact = {}
    for identity in sorted(records):
        previous = exact.get(records[identity]['exact_hash'])
        if previous is None:
            exact[records[identity]['exact_hash']] = identity
        elif constraints.can_union(sets, previous, identity, forbidden):
            sets.union(previous, identity)
        else:
            suppressed.append({'left_id': previous, 'right_id': identity, 'reason': 'different_constraint'})
    signatures = {identity: similarity.shingles(records[identity]['normalized'], run['shingle_size']) for identity in sorted(records)}
    for left, right in similarity.candidates(signatures, run['threshold_numerator'], run['threshold_denominator']):
        if records[left]['exact_hash'] == records[right]['exact_hash']:
            continue
        if not similarity.matches(signatures[left], signatures[right], run['threshold_numerator'], run['threshold_denominator']):
            continue
        if constraints.can_union(sets, left, right, forbidden):
            sets.union(left, right)
        else:
            suppressed.append({'left_id': left, 'right_id': right, 'reason': 'different_constraint'})
    assignment = {}
    groups = []
    for members in sets.groups():
        cluster_id = 'isolation-' + digest(sorted(members))
        for identity in members:
            assignment[identity] = cluster_id
        groups.append({'cluster_id': cluster_id, 'members': sorted(members), 'from_run': all(identity in run_members for identity in members)})
    groups.sort(key=lambda group: group['cluster_id'])
    return {
        'method': 'pinned_run_with_parameter_completion',
        'run_id': run['run_id'],
        'run_digest': run['digest'],
        'run_fresh': run['fresh'],
        'run_created_at': run['created_at'],
        'run_corpus_revision': run['corpus_revision'],
        'run_predates_cutoff': run['created_at'] <= cutoff,
        'run_parameters': {'numerator': run['threshold_numerator'], 'denominator': run['threshold_denominator'], 'shingle_size': run['shingle_size']},
        'documents': len(universe),
        'clusters': len(groups),
        'derived_members': sorted(identity for identity in universe if identity not in run_members),
        'suppressed': suppressed,
        'groups': groups,
        'assignment': assignment,
    }


def _candidate_snapshot(db, options):
    candidates = []
    excluded = []
    for document in documents.listing(db, limit=10000):
        if document['state'] not in REVIEWABLE:
            excluded.append({'document_id': document['document_id'], 'reason': 'state_not_reviewable', 'state': document['state']})
            continue
        if not document['is_head']:
            excluded.append({'document_id': document['document_id'], 'reason': 'not_current_head'})
            continue
        decided = db.one('SELECT * FROM review_events WHERE document_id=? AND decision IN (?,?) ORDER BY event_id DESC LIMIT 1', (document['document_id'], 'accept', 'reject'))
        if decided:
            excluded.append({'document_id': document['document_id'], 'reason': 'already_human_decided', 'decision': decided['decision'], 'decided_at': instant(decided['at'])})
            continue
        claim = _active_claim(db, document['document_id'])
        if claim:
            excluded.append({'document_id': document['document_id'], 'reason': 'already_claimed', 'claim': {'owner': claim['owner'], 'generation': claim['generation'], 'expires_at': claim['expires_at']}})
            continue
        raw_minutes = document['metadata'].get('minutes')
        if type(raw_minutes) is int and 1 <= raw_minutes <= 100000:
            minutes, minutes_source = raw_minutes, 'metadata'
        else:
            minutes, minutes_source = options['default_minutes'], 'default'
        summary = risk.rule_summary(document['findings'])
        candidates.append({
            'document_id': document['document_id'],
            'document_revision': document['revision'],
            'state': document['state'],
            'source_id': document['source_id'],
            'created_at': document['created_at'],
            'text': document['normalized'],
            'minutes': minutes,
            'minutes_source': minutes_source,
            'rule': summary,
            'rule_stratum': risk.rule_stratum(summary),
        })
    candidates.sort(key=lambda row: (row['created_at'], row['document_id']))
    return candidates, excluded


def plan(db, **values):
    options = settings(values)
    run = clusters.get(db, options['run_id'])
    training, label_exclusions = _label_candidates(db, options['cutoff'])
    candidates, candidate_exclusions = _candidate_snapshot(db, options)
    universe = sorted({row['document_id'] for row in training} | {row['document_id'] for row in candidates})
    partition = _partition(db, run, universe, options['cutoff'])
    for row in training + candidates:
        row['cluster_id'] = partition['assignment'][row['document_id']]
    model = risk.fit(training, options['model'])
    lexical_index, lexical_idf = risk.lexical_profile(training)
    for row in training:
        row['lexical'] = risk.lexical_score(lexical_idf, row['text'])
        if row['minutes'] is None:
            row['minutes'] = options['default_minutes']
            row['minutes_source'] = 'default'
    for row in candidates:
        row['lexical'] = risk.lexical_score(lexical_idf, row['text'])
        row['prediction'] = risk.predict(model, row['text'])
        row['score'] = {
            'probability_reject': row['prediction']['probability'],
            'uncertainty_bits': row['prediction']['uncertainty'],
            'rule_priority': row['rule']['priority'],
        }
    backtest = risk.evaluate(training, options['budget_minutes'], options['model'], options['thresholds'], options['seed'])
    enabled = backtest['verdict']['worth_enabling']
    strategy = 'text_model_active' if enabled else 'rule_priority_isolated'
    selection = _select(candidates, options, strategy)
    evidence = {
        'format': 'review-round-plan',
        'version': 1,
        'options': options,
        'request_digest': _request_digest(options),
        'cutoff': options['cutoff'],
        'corpus_revision': db.revision(),
        'dedup': {key: value for key, value in partition.items() if key != 'assignment'},
        'training': {
            'documents': len(training),
            'labels': {risk.POSITIVE: sum(row['label'] for row in training), risk.NEGATIVE: len(training) - sum(row['label'] for row in training)},
            'decisions': [{key: row[key] for key in ('document_id', 'event_id', 'decision', 'decided_at', 'cluster_id', 'minutes', 'minutes_source', 'lexical')} for row in training],
            'excluded_decisions': label_exclusions,
            'cutoff_enforced': True,
            'semantics': 'only final human accept/reject decisions on the exact current document version at or before the cutoff are used; rule findings are never labels',
        },
        'lexical_baseline': {
            'definition': 'mean inverse-document-frequency of distinct tokens, fitted on the training texts only; no labels involved',
            'training_documents': len(training),
            'vocabulary_terms': len(lexical_index),
        },
        'model': model,
        'backtest': backtest,
        'strategy': {
            'name': strategy,
            'model_enabled': enabled,
            'decision': backtest['verdict'],
            'random_exploration_seed': options['seed'],
            'semantics': 'When forward evidence does not clear the configured thresholds the round falls back to deterministic rule priority, still one document per dedup cluster.',
        },
        'candidates': _candidate_evidence(candidates, selection),
        'candidate_exclusions': candidate_exclusions,
        'selection': selection,
        'queue': selection['chosen'],
    }
    return evidence


def _candidate_evidence(candidates, selection):
    rows = []
    chosen = {item['document_id']: item for item in selection['chosen']}
    for candidate in candidates:
        row = {
            'document_id': candidate['document_id'],
            'document_revision': candidate['document_revision'],
            'source_id': candidate['source_id'],
            'cluster_id': candidate['cluster_id'],
            'minutes': candidate['minutes'],
            'minutes_source': candidate['minutes_source'],
            'rule': candidate['rule'],
            'prediction': {key: candidate['prediction'][key] for key in ('probability', 'uncertainty', 'fallback', 'explanation', 'tokens', 'known_terms', 'unknown_terms')},
            'selected': candidate['document_id'] in chosen,
            'phase': chosen[candidate['document_id']]['phase'] if candidate['document_id'] in chosen else None,
            'reasons': chosen[candidate['document_id']]['reasons'] if candidate['document_id'] in chosen else selection['rejected'].get(candidate['document_id'], ['not_evaluated']),
        }
        rows.append(row)
    rows.sort(key=lambda row: row['document_id'])
    return rows


def _select(candidates, options, strategy):
    """Deterministic floor -> exploration -> ranking selection under constraints."""
    budget = options['budget_minutes']
    floor_sources = options['source_floor']
    floor_documents = options['floor_documents']
    used_clusters = {}
    phase_of = {}
    chosen = []
    rejected = {}
    selected_sources = {}
    used_minutes = 0

    def take(row, phase, reasons):
        nonlocal used_minutes
        chosen.append({
            'document_id': row['document_id'],
            'document_revision': row['document_revision'],
            'cluster_id': row['cluster_id'],
            'source_id': row['source_id'],
            'minutes': row['minutes'],
            'phase': phase,
            'score': row['score'],
            'reasons': reasons,
            'cumulative_minutes': used_minutes + row['minutes'],
        })
        used_clusters[row['cluster_id']] = row['document_id']
        selected_sources[row['source_id']] = selected_sources.get(row['source_id'], 0) + 1
        phase_of[row['document_id']] = phase
        used_minutes += row['minutes']

    def reject(row, reason):
        rejected.setdefault(row['document_id'], []).append(reason)

    # Floor picks are chosen first so an infeasible floor is a loud conflict,
    # never a silently dropped constraint.
    floor_picks = []
    for source_id in floor_sources:
        available = sorted(
            (row for row in candidates if row['source_id'] == source_id),
            key=lambda row: (row['minutes'], digest([options['seed'], source_id, row['document_id']]), row['document_id']),
        )
        picks = []
        for row in available:
            if len(picks) == floor_documents:
                break
            if any(item['cluster_id'] == row['cluster_id'] for item in picks):
                continue
            picks.append(row)
        if len(picks) < floor_documents:
            fail(
                'round_source_floor_infeasible',
                'Source floor cannot be met with distinct dedup clusters',
                source_id=source_id,
                requirement=floor_documents,
                available_documents=len(available),
                available_clusters=len({row['cluster_id'] for row in available}),
            )
        floor_picks.extend(picks)
    floor_minutes = sum(item['minutes'] for item in floor_picks)
    if floor_minutes > budget:
        fail(
            'round_source_floor_infeasible',
            'Source floor alone exceeds the minute budget',
            requirement_minutes=floor_minutes,
            budget_minutes=budget,
            source_floor=floor_sources,
            conflict='the floor cannot be dropped or silently trimmed; raise the budget or lower floor_documents',
        )
    floor_cluster_usage = {}
    for row in floor_picks:
        if row['cluster_id'] in floor_cluster_usage:
            fail(
                'round_source_floor_infeasible',
                'Floor documents from different sources share one dedup cluster',
                cluster_id=row['cluster_id'],
                documents=[floor_cluster_usage[row['cluster_id']], row['document_id']],
                conflict='one dedup cluster may contribute at most one document per round',
            )
        floor_cluster_usage[row['cluster_id']] = row['document_id']
    for row in floor_picks:
        take(row, 'source_floor', [{'code': 'source_floor', 'source_id': row['source_id'], 'requirement_documents': floor_documents}])

    exploration_target = min(budget - floor_minutes, int(math.floor(budget * options['exploration_fraction'])))
    exploration_spent = 0
    exploration_ids = []
    exploration_skips = {}
    if exploration_target > 0:
        ordered = sorted(
            candidates,
            key=lambda row: (digest([options['seed'], options['round_id'], 'exploration', row['document_id']]), row['document_id']),
        )
        # When the share cannot be spent because selected clusters are already
        # covered, the shortfall is reported by exploration.spent vs target.
        for row in ordered:
            if row['document_id'] in phase_of:
                continue
            if row['cluster_id'] in used_clusters:
                exploration_skips.setdefault(row['document_id'], 'cluster_already_selected')
                continue
            if exploration_spent + row['minutes'] > exploration_target:
                exploration_skips.setdefault(row['document_id'], 'exploration_target_too_small')
                continue
            take(row, 'exploration', [{
                'code': 'fixed_seed_exploration',
                'seed': options['seed'],
                'cluster_id': row['cluster_id'],
                'minutes_spent': exploration_spent + row['minutes'],
                'minutes_target': exploration_target,
            }])
            exploration_ids.append(row['document_id'])
            exploration_spent += row['minutes']

    def priority(row, counts):
        if strategy == 'text_model_active':
            base = row['prediction']['uncertainty']
        else:
            base = row['rule']['priority']
        coverage = 1.0 / (1.0 + counts.get(row['source_id'], 0))
        return base * (1.0 + options['coverage_weight'] * coverage)

    while True:
        counts = dict(selected_sources)
        best = None
        remaining = budget - used_minutes
        for row in candidates:
            if row['document_id'] in phase_of or row['cluster_id'] in used_clusters:
                continue
            if row['minutes'] > remaining:
                continue
            value = priority(row, counts)
            key = (value / row['minutes'], row['prediction']['uncertainty'], row['document_id'])
            if best is None or key > best[0]:
                best = (key, row)
        if best is None:
            break
        row = best[1]
        take(row, 'ranking', [{
            'code': 'ranking_score',
            'score_per_minute': best[0][0],
            'base': priority(row, counts),
            'base_source': 'uncertainty' if strategy == 'text_model_active' else 'rule_priority',
            'coverage_factor': 1.0 + options['coverage_weight'] / (1.0 + counts.get(row['source_id'], 0)),
            'uncertainty_bits': row['prediction']['uncertainty'],
            'rule_priority': row['rule']['priority'],
            'uncertainty_tiebreak': True,
        }])

    for row in candidates:
        if row['document_id'] in phase_of:
            continue
        if row['cluster_id'] in used_clusters:
            reject(row, 'cluster_already_selected')
        elif row['minutes'] > budget - used_minutes:
            reject(row, 'minute_budget')
        else:
            reject(row, 'lower_ranked_than_selected')
        if row['document_id'] in exploration_skips:
            reject(row, exploration_skips[row['document_id']])
    rejected = {identity: reasons for identity, reasons in rejected.items() if identity not in phase_of}
    return {
        'strategy': strategy,
        'budget_minutes': budget,
        'used_minutes': used_minutes,
        'remaining_minutes': budget - used_minutes,
        'exploration': {'target_minutes': exploration_target, 'spent_minutes': exploration_spent, 'documents': exploration_ids, 'fraction': options['exploration_fraction'], 'seed': options['seed']},
        'source_floor': {'sources': options['source_floor'], 'documents_each': floor_documents, 'satisfied': all(selected_sources.get(source, 0) >= floor_documents for source in options['source_floor']), 'selected_by_source': dict(sorted(selected_sources.items()))},
        'chosen': chosen,
        'rejected': rejected,
        'counts': {'documents': len(chosen), 'sources': dict(sorted(selected_sources.items())), 'clusters': len(used_clusters)},
        'constraints': {
            'one_per_cluster': True,
            'minute_budget_respected': used_minutes <= budget,
            'source_floor_checked_before_spending': True,
            'source_floor_unsatisfied_raises_conflict': 'round_source_floor_infeasible',
        },
    }


def create(db, **values):
    options = settings(values)
    request = _request_digest(options)
    existing = db.one('SELECT * FROM review_rounds WHERE round_id=?', (options['round_id'],))
    if existing:
        if existing['request_digest'] != request:
            fail('round_conflict', 'Round identifier is bound to a different request', round_id=options['round_id'])
        return get(db, options['round_id'])
    evidence = plan(db, **values)
    if db.revision() != evidence['corpus_revision']:
        fail('round_corpus_changed', 'Corpus changed while planning the round')
    sha = digest(evidence)
    with db.transaction():
        previous = db.one('SELECT * FROM review_rounds WHERE round_id=?', (options['round_id'],))
        if previous:
            if previous['request_digest'] != request:
                fail('round_conflict', 'Round identifier is bound to a different request', round_id=options['round_id'])
            return get(db, options['round_id'])
        db.execute(
            "INSERT INTO review_rounds VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (options['round_id'], 'active', options['run_id'], options['cutoff'], options['budget_minutes'], options['seed'], canonical(options), request, canonical(evidence), sha, evidence['corpus_revision'], db.clock()),
        )
        for ordinal, item in enumerate(evidence['queue']):
            db.execute('INSERT INTO review_round_queue VALUES(?,?,?,?,?)', (options['round_id'], item['document_id'], ordinal, item['document_revision'], item['cluster_id']))
        db.execute('INSERT INTO review_round_events(round_id,action,body,at) VALUES(?,?,?,?)', (options['round_id'], 'created', canonical({'digest': sha, 'documents': [item['document_id'] for item in evidence['queue']]}), db.clock()))
        db.log('review.round_created', {'round_id': options['round_id'], 'digest': sha, 'documents': len(evidence['queue']), 'strategy': evidence['strategy']['name']})
        return get(db, options['round_id'])


def get(db, round_id):
    identifier(round_id)
    row = db.one('SELECT * FROM review_rounds WHERE round_id=?', (round_id,))
    if row is None:
        fail('round_missing', 'Review round does not exist')
    stored = json.loads(row.pop('evidence'))
    if digest(stored) != row['digest']:
        fail('round_corrupt', 'Round evidence digest differs')
    row['options'] = json.loads(row.pop('definition'))
    row['queue'] = db.all('SELECT document_id,document_revision,cluster_id,ordinal FROM review_round_queue WHERE round_id=? ORDER BY ordinal', (round_id,))
    row['events'] = db.all('SELECT action,body,at FROM review_round_events WHERE round_id=? ORDER BY event_id', (round_id,))
    for event in row['events']:
        event['body'] = json.loads(event['body'])
    row['evidence'] = stored
    return row


def _rebuild_candidates(stored):
    rows = []
    for item in stored['candidates']:
        rows.append({
            'document_id': item['document_id'],
            'document_revision': item['document_revision'],
            'source_id': item['source_id'],
            'cluster_id': item['cluster_id'],
            'minutes': item['minutes'],
            'rule': item['rule'],
            'prediction': {**item['prediction'], 'top_terms': []},
            'score': {
                'probability_reject': item['prediction']['probability'],
                'uncertainty_bits': item['prediction']['uncertainty'],
                'rule_priority': item['rule']['priority'],
            },
        })
    return rows


def recompute_selection(stored):
    """Re-run the selector from the frozen evidence alone.

    This proves the stored queue is reproducible from the saved candidate
    scores, costs and constraints, without touching the live corpus.
    """
    rows = _rebuild_candidates(stored)
    options = stored['options']
    replayed = _select(rows, options, stored['strategy']['name'])
    stored_ids = [item['document_id'] for item in stored['queue']]
    replay_ids = [item['document_id'] for item in replayed['chosen']]
    return {
        'matches_stored_queue': stored_ids == replay_ids,
        'stored_queue': stored_ids,
        'recomputed_queue': replay_ids,
        'used_minutes': replayed['used_minutes'],
        'budget_minutes': replayed['budget_minutes'],
        'within_budget': replayed['used_minutes'] <= replayed['budget_minutes'],
        'one_per_cluster': len({item['cluster_id'] for item in replayed['chosen']}) == len(replayed['chosen']),
        'exploration': replayed['exploration'],
        'source_floor': replayed['source_floor'],
    }


def evidence(db, round_id):
    """Stored plan plus independent recomputation.

    `stored_evidence_valid` checks the frozen digest. `selection_recheck`
    re-runs the selector from stored evidence only. `replay_now` re-plans the
    same request against the current corpus and human history, so a changed
    corpus or new decisions show up as an explicit digest difference.
    """
    current = get(db, round_id)
    stored = current['evidence']
    stored_digest = digest(stored)
    recheck = recompute_selection(stored)
    replay = None
    try:
        replay = plan(db, **current['options'])
    except GateError as exc:
        replay = {'error': exc.as_dict()}
    replay_digest = digest(replay) if 'error' not in replay else None
    return {
        'round_id': round_id,
        'state': current['state'],
        'stored_digest': stored_digest,
        'stored_evidence_valid': stored_digest == current['digest'],
        'selection_recheck': recheck,
        'replay_now': {'digest': replay_digest, 'matches_stored': replay_digest == stored_digest, 'error': replay.get('error') if 'error' in replay else None},
        'queue': current['queue'],
        'options': current['options'],
        'evidence': stored,
    }


def validate(db, round_id):
    """Report why a frozen round no longer applies, without mutating queue."""
    current = get(db, round_id)
    selected = {item['document_id']: item for item in current['evidence']['queue']}
    invalid = []
    for item in current['queue']:
        document = documents.get(db, item['document_id'])
        reasons = []
        versions = documents.versions(db, document['source_id'], document['external_id'])
        newest = max(row['version'] for row in versions)
        if document['version'] < newest:
            reasons.append({'code': 'candidate_version_superseded', 'frozen_version': document['version'], 'newest_version': newest})
        if document['revision'] != item['document_revision']:
            reasons.append({'code': 'document_revision_changed', 'frozen': item['document_revision'], 'current': document['revision']})
        if document['state'] == 'withdrawn':
            reasons.append({'code': 'document_withdrawn'})
        elif document['state'] not in REVIEWABLE:
            reasons.append({'code': 'document_state', 'current': document['state']})
        claim = _active_claim(db, item['document_id'])
        if claim:
            reasons.append({'code': 'currently_claimed', 'claim': {'owner': claim['owner'], 'generation': claim['generation'], 'expires_at': claim['expires_at']}})
        history = db.one('SELECT * FROM review_events WHERE document_id=? ORDER BY event_id DESC LIMIT 1', (document['document_id'],))
        if history and instant(history['at']) > instant(current['created_at']):
            reasons.append({'code': 'reviewed_after_round', 'event_id': history['event_id'], 'decision': history['decision'], 'at': history['at']})
        if reasons:
            invalid.append({'document_id': item['document_id'], 'frozen_revision': item['document_revision'], 'reasons': reasons})
    return {
        'round_id': round_id,
        'state': current['state'],
        'valid': not invalid,
        'invalid': invalid,
        'queue': current['queue'],
        'selected_documents': sorted(selected),
        'semantics': 'A frozen round is advisory. Invalid queue items are reported, not re-filled; create a new round to spend the recovered minutes.',
    }


def close(db, round_id):
    with db.transaction():
        current = db.one('SELECT * FROM review_rounds WHERE round_id=?', (round_id,))
        if current is None:
            fail('round_missing', 'Review round does not exist')
        if current['state'] == 'closed':
            return get(db, round_id)
        db.execute("UPDATE review_rounds SET state='closed' WHERE round_id=?", (round_id,))
        db.execute('INSERT INTO review_round_events(round_id,action,body,at) VALUES(?,?,?,?)', (round_id, 'closed', canonical({}), db.clock()))
        db.log('review.round_closed', {'round_id': round_id})
        return get(db, round_id)
