"""Text risk model, calibration, and time-forward historical check.

Only human review decisions made on or before a round cutoff are used as
labels. Rule findings are used for the naive baseline and as a ranking
baseline, never as training truth. Vocabulary, weights and Platt calibration
are fitted inside the training portion only.
"""

import math

from corpus_gate.text.normalize import tokens

POSITIVE = 'reject'
NEGATIVE = 'accept'


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def entropy(probability):
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def terms_of(text):
    return [token['text'].casefold() for token in tokens(text or '')]


def vocabularies(texts, minimum_document_frequency=1, maximum=2000):
    counts = {}
    for text in texts:
        for term in sorted(set(terms_of(text))):
            counts[term] = counts.get(term, 0) + 1
    eligible = [(term, count) for term, count in counts.items() if count >= minimum_document_frequency]
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))[:maximum]
    return [term for term, _ in ranked]


def lexical_profile(training):
    """Label-free TF-IDF-ish reference fitted on training texts only.

    Returns a document-frequency map and the idf copy, so a replay in another
    process can reproduce the naive lexical score.
    """
    counts = {}
    for row in training:
        for term in sorted(set(terms_of(row['text']))):
            counts[term] = counts.get(term, 0) + 1
    total = len(training)
    tenure = max(total, 1)
    idf = {term: math.log((1.0 + tenure) / (1.0 + count)) + 1.0 for term, count in sorted(counts.items())}
    return counts, idf


def lexical_score(idf, text):
    terms = sorted(set(terms_of(text)))
    shared = [idf[term] for term in terms if term in idf]
    if not terms:
        return 0.0
    return sum(shared) / len(terms)


def _feature_row(text, index):
    return sorted({index[term] for term in terms_of(text) if term in index})


def _fit_weights(texts, labels, vocabulary, options):
    index = {term: position for position, term in enumerate(vocabulary)}
    rows = [_feature_row(text, index) for text in texts]
    positives = sum(labels)
    negatives = len(labels) - positives
    bias = math.log((positives + 1) / (negatives + 1))
    weights = [0.0] * len(vocabulary)
    l2 = options['l2']
    rate = options['learning_rate']
    for _ in range(options['epochs']):
        gradient = [0.0] * len(vocabulary)
        bias_gradient = 0.0
        for row, label in zip(rows, labels):
            probability = sigmoid(bias + sum(weights[position] for position in row))
            error = probability - label
            for position in row:
                gradient[position] += error
            bias_gradient += error
        count = len(rows)
        for position in range(len(weights)):
            weights[position] -= rate * (gradient[position] / count + l2 * weights[position])
        bias -= rate * (bias_gradient / count)
    return weights, bias


def _fit_calibration(margins, labels, rate=0.2, epochs=400):
    """Positivity-constrained Platt fit via projected gradient."""
    if len(set(labels)) < 2:
        return 1.0, 0.0
    candidates = [(1.0, 0.0), (0.5, 0.0), (1.0, -1.0)]
    best = None
    for initial_scale, initial_shift in candidates:
        scale, shift = initial_scale, initial_shift
        for _ in range(epochs):
            scale_gradient = 0.0
            shift_gradient = 0.0
            for margin, label in zip(margins, labels):
                error = sigmoid(scale * margin + shift) - label
                scale_gradient += error * margin
                shift_gradient += error
            count = len(margins)
            scale = max(scale - rate * scale_gradient / count, 1e-6)
            shift -= rate * shift_gradient / count
        loss = 0.0
        for margin, label in zip(margins, labels):
            probability = sigmoid(scale * margin + shift)
            bounded = min(max(probability, 1e-12), 1.0 - 1e-12)
            loss -= label * math.log(bounded) + (1.0 - label) * math.log(1.0 - bounded)
        if best is None or loss < best[0]:
            best = (loss, scale, shift)
    return best[1], best[2]


def fit(training, options):
    """Fit vocabulary, weights and calibration from `training` only.

    Each training record carries document_id, text, label (0 accept / 1 reject)
    and decided_at. The latest 30% by decision time calibrates; earlier records
    fit weights. A single-class or empty-vocabulary training set returns an
    explicit constant-probability fallback instead of inventing a boundary.
    """
    texts = [row['text'] for row in training]
    labels = [row['label'] for row in training]
    vocabulary = vocabularies(texts, options['min_document_frequency'], options['max_vocabulary'])
    positives = sum(labels)
    negatives = len(labels) - positives
    evidence = {
        'type': 'l2_logistic_binary_text',
        'positive_label': POSITIVE,
        'feature_definition': 'binary_presence_of_casefolded_tokens_in_training_vocabulary',
        'hyperparameters': dict(options),
        'vocabulary_size': len(vocabulary),
        'training_documents': len(training),
        'training_labels': {POSITIVE: positives, NEGATIVE: negatives},
        'training_ids': [row['document_id'] for row in sorted(training, key=lambda item: item['document_id'])],
        'fallbacks': [],
    }
    if positives == 0 or negatives == 0:
        probability = (positives + 1) / (len(training) + 2)
        evidence.update({
            'status': 'constant_single_class',
            'constant_probability': probability,
            'intercept': math.log(probability / (1.0 - probability)),
            'weights': [],
            'vocabulary': [],
            'calibration': {'method': 'not_applicable', 'reason': 'single_class_training_labels'},
        })
        evidence['fallbacks'].append({
            'code': 'single_class_labels',
            'action': 'constant smoothed prior for every candidate; ranking falls back to uncertainty then rule priority',
            'probability': probability,
        })
        return evidence
    if not vocabulary:
        probability = (positives + 1) / (len(training) + 2)
        evidence.update({
            'status': 'constant_no_vocabulary',
            'constant_probability': probability,
            'intercept': math.log(probability / (1.0 - probability)),
            'weights': [],
            'vocabulary': [],
            'calibration': {'method': 'not_applicable', 'reason': 'empty_text_in_training'},
        })
        evidence['fallbacks'].append({
            'code': 'no_vocabulary',
            'action': 'constant smoothed prior; texts had no usable tokens during training',
            'probability': probability,
        })
        return evidence

    ordered = sorted(training, key=lambda item: (item['decided_at'], item['document_id']))
    calibration_size = round(len(ordered) * 0.3) if len(ordered) >= 8 else 0
    fit_rows = ordered[:len(ordered) - calibration_size]
    calibration_rows = ordered[len(ordered) - calibration_size:] if calibration_size else []
    weights, bias = _fit_weights([row['text'] for row in fit_rows], [row['label'] for row in fit_rows], vocabulary, options)
    index = {term: position for position, term in enumerate(vocabulary)}

    def margin(text):
        return bias + sum(weights[position] for position in _feature_row(text, index))

    calibration = {'method': 'identity', 'scale': 1.0, 'shift': 0.0, 'documents': len(calibration_rows), 'reason': 'calibration_slice_unavailable'}
    calibration_labels = [row['label'] for row in calibration_rows]
    if calibration_rows and 0 < sum(calibration_labels) < len(calibration_labels):
        scale, shift = _fit_calibration([margin(row['text']) for row in calibration_rows], calibration_labels)
        if math.isfinite(scale) and math.isfinite(shift) and scale > 0:
            calibration = {'method': 'platt', 'scale': scale, 'shift': shift, 'documents': len(calibration_rows), 'reason': 'time_forward_holdout_inside_training'}
        else:
            calibration['reason'] = 'calibration_nonfinite'
    evidence.update({
        'status': 'fitted',
        'vocabulary': vocabulary,
        'weights': weights,
        'intercept': bias,
        'calibration': calibration,
        'fit': {
            'documents': len(fit_rows),
            'ids': [row['document_id'] for row in fit_rows],
            'labels': {POSITIVE: sum(row['label'] for row in fit_rows), NEGATIVE: len(fit_rows) - sum(row['label'] for row in fit_rows)},
        },
        'calibration_documents': {
            'documents': len(calibration_rows),
            'ids': [row['document_id'] for row in calibration_rows],
        },
    })
    if calibration_rows and calibration['method'] == 'identity' and calibration['reason'] == 'calibration_slice_unavailable':
        evidence['fallbacks'].append({'code': 'calibration_unavailable', 'action': 'identity link; probabilities remain raw margins'})
    elif calibration['method'] == 'identity' and calibration_rows:
        evidence['fallbacks'].append({'code': 'calibration_single_class', 'action': 'identity link; holdout had one class'})
    evidence['calibration']['reason'] = calibration['reason']
    return evidence


def predict(model, text):
    """Return probability of reject plus enough explanation to audit it."""
    terms = terms_of(text)
    if model['status'] != 'fitted':
        return {
            'probability': model['constant_probability'],
            'uncertainty': entropy(model['constant_probability']),
            'fallback': model['status'],
            'explanation': 'no discriminative text model; constant smoothed prior from training labels',
            'tokens': len(terms),
            'known_terms': 0,
            'unknown_terms': len(terms),
            'top_terms': [],
        }
    index = {term: position for position, term in enumerate(model['vocabulary'])}
    present = sorted({term for term in terms if term in index})
    unknown = sorted({term for term in terms if term not in index})
    margin = model['intercept'] + sum(model['weights'][index[term]] for term in present)
    calibrated = model['calibration']['scale'] * margin + model['calibration']['shift']
    probability = sigmoid(calibrated)
    fallback = None
    explanation = 'regularized logistic text score'
    if not terms:
        fallback = 'empty_text'
        explanation = 'no tokens after normalization; score equals the intercept/calibration prior'
    elif not present:
        fallback = 'unknown_vocabulary'
        explanation = 'every token is outside the training vocabulary; score equals the intercept/calibration prior'
    elif unknown:
        explanation = 'some tokens are outside the training vocabulary and contribute no weight'
    top = sorted(
        ((term, model['weights'][index[term]]) for term in present),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:8]
    return {
        'probability': probability,
        'uncertainty': entropy(probability),
        'fallback': fallback,
        'explanation': explanation,
        'tokens': len(terms),
        'known_terms': len(present),
        'unknown_terms': len(unknown),
        'top_terms': [{'term': term, 'weight': weight} for term, weight in top],
    }


def rule_summary(findings):
    hard = sum(1 for finding in findings if finding['severity'] == 'reject')
    warnings = len(findings) - hard
    score = 1.0 if hard else 0.5 if warnings else 0.0
    return {
        'hard_findings': hard,
        'quarantine_findings': warnings,
        'codes': sorted({finding['code'] for finding in findings}),
        'priority': score,
    }


def rule_stratum(summary):
    if summary['hard_findings']:
        return 'hard'
    if summary['quarantine_findings']:
        return 'quarantine'
    return 'clean'


def rule_rank(document):
    summary = document['rule']
    return (-summary['hard_findings'], -summary['quarantine_findings'], document['created_at'], document['document_id'])


def model_rank(document):
    return (-document['prediction']['probability'], document['created_at'], document['document_id'])


def lexical_rank(document):
    """Plain text-frequency score with no labels; the naive lexical baseline."""
    return (-document['lexical'], document['created_at'], document['document_id'])


def stratum_rates(training):
    counts = {}
    for row in training:
        stratum = row['stratum']
        entry = counts.setdefault(stratum, [0, 0])
        entry[0] += row['label']
        entry[1] += 1
    positives = sum(row['label'] for row in training)
    overall = (positives + 0.5) / (len(training) + 1.0)
    return {stratum: (values[0] + 0.5) / (values[1] + 1.0) for stratum, values in counts.items()}, overall


def _feasible_subset(documents, budget_minutes, ranker):
    """Sort by rank, then greedily admit anything that still fits.

    A skipped costly document no longer closes the budget to cheaper ones.
    """
    ordered = sorted(documents, key=ranker)
    selected = []
    skipped = []
    used = 0
    for document in ordered:
        if used + document['minutes'] > budget_minutes:
            skipped.append({'document_id': document['document_id'], 'reason': 'minute_budget', 'minutes': document['minutes'], 'remaining': budget_minutes - used})
            continue
        selected.append(document)
        used += document['minutes']
    return selected, skipped, used


def _simulate(documents, ranker, budget_minutes, dedupe_clusters):
    candidates, skipped, used = _feasible_subset(documents, budget_minutes, ranker)
    selected = []
    clusters = {}
    repeated = set()
    for document in candidates:
        if document['cluster_id'] in clusters:
            if dedupe_clusters:
                skipped.append({'document_id': document['document_id'], 'reason': 'cluster_already_selected', 'cluster_id': document['cluster_id'], 'by': clusters[document['cluster_id']]})
                continue
            repeated.add(document['document_id'])
        else:
            clusters[document['cluster_id']] = document['document_id']
        selected.append(document)
    if dedupe_clusters:
        used = sum(document['minutes'] for document in selected)
    hits = sum(document['label'] for document in selected)
    rejects = sum(document['label'] for document in documents)
    revisit_minutes = sum(document['minutes'] for document in selected if document['document_id'] in repeated)
    return {
        'selected_documents': [document['document_id'] for document in selected],
        'selected': len(selected),
        'hits': hits,
        'rejects_available': rejects,
        'minutes': used,
        'budget_minutes': budget_minutes,
        'precision': hits / len(selected) if selected else None,
        'recall': hits / rejects if rejects else None,
        'distinct_clusters': len(clusters),
        'revisited_documents': len(repeated),
        'revisited_minutes': revisit_minutes,
        'one_per_cluster_enforced': bool(dedupe_clusters),
        'skipped': skipped,
    }


def evaluate(training, budget_minutes, options, thresholds, seed):
    """Forward-time, cluster-isolated backtest of rules versus the text model.

    Each fold trains only on decisions strictly before the fold's decision
    time and tests on that time's decisions. Test documents sharing a
    near-duplicate cluster with any training document are dropped before
    scoring. Both Brier and budget simulations use the human labels.
    """
    times = sorted({row['decided_at'] for row in training})
    folds = []
    for stamp in times[1:]:
        train_rows = [row for row in training if row['decided_at'] < stamp]
        test_rows = [row for row in training if row['decided_at'] == stamp]
        if not train_rows or not test_rows:
            continue
        train_clusters = {row['cluster_id'] for row in train_rows}
        isolated_out = [row['document_id'] for row in test_rows if row['cluster_id'] in train_clusters]
        kept = [row for row in test_rows if row['cluster_id'] not in train_clusters]
        if not kept:
            continue
        representatives = {}
        for row in kept:
            key = row['cluster_id']
            if key not in representatives or row['document_id'] < representatives[key]['document_id']:
                representatives[key] = row
        deduped = sorted(representatives.values(), key=lambda row: row['document_id'])
        fold_model = fit(train_rows, options)
        rates, overall = stratum_rates(train_rows)
        _, fold_idf = lexical_profile(train_rows)
        candidates = []
        for row in kept:
            prediction = predict(fold_model, row['text'])
            candidates.append({
                'document_id': row['document_id'],
                'label': row['label'],
                'cluster_id': row['cluster_id'],
                'minutes': row['minutes'],
                'created_at': row['created_at'],
                'rule': row['rule'],
                'lexical': lexical_score(fold_idf, row['text']),
                'prediction': prediction,
                'rule_probability': rates.get(row['stratum'], overall),
            })
        clustered = [row for row in candidates if any(item['document_id'] == row['document_id'] for item in deduped)]
        naive = _simulate(candidates, rule_rank, budget_minutes, False)
        lexical_naive = _simulate(clustered, lexical_rank, budget_minutes, True)
        rule_isolated = _simulate(clustered, rule_rank, budget_minutes, True)
        model_isolated = _simulate(clustered, model_rank, budget_minutes, True)
        brier_model = sum((row['prediction']['probability'] - row['label']) ** 2 for row in candidates) / len(candidates)
        brier_rule = sum((row['rule_probability'] - row['label']) ** 2 for row in candidates) / len(candidates)
        folds.append({
            'decision_time': stamp,
            'training_documents': len(train_rows),
            'training_labels': {POSITIVE: sum(row['label'] for row in train_rows), NEGATIVE: len(train_rows) - sum(row['label'] for row in train_rows)},
            'test_documents': len(candidates),
            'test_labels': {POSITIVE: sum(row['label'] for row in candidates), NEGATIVE: len(candidates) - sum(row['label'] for row in candidates)},
            'test_clusters': len(representatives),
            'isolated_duplicates_removed': isolated_out,
            'cluster_isolated': True,
            'model_status': fold_model['status'],
            'model_calibration': fold_model['calibration'],
            'brier': {'model': brier_model, 'rule_stratum': brier_rule},
            'strategies': {'rule_naive': naive, 'lexical_naive': lexical_naive, 'rule_isolated': rule_isolated, 'model_isolated': model_isolated},
        })
    total_test = sum(fold['test_documents'] for fold in folds)
    total_rejects = sum(fold['test_labels'][POSITIVE] for fold in folds)
    total_accepts = total_test - total_rejects
    if folds:
        brier_model = sum(fold['brier']['model'] * fold['test_documents'] for fold in folds) / total_test
        brier_rule = sum(fold['brier']['rule_stratum'] * fold['test_documents'] for fold in folds) / total_test
        aggregate = {}
        for strategy in ('rule_naive', 'lexical_naive', 'rule_isolated', 'model_isolated'):
            selected = sum(fold['strategies'][strategy]['selected'] for fold in folds)
            hits = sum(fold['strategies'][strategy]['hits'] for fold in folds)
            minutes = sum(fold['strategies'][strategy]['minutes'] for fold in folds)
            available = sum(fold['strategies'][strategy]['rejects_available'] for fold in folds)
            aggregate[strategy] = {
                'selected': selected,
                'hits': hits,
                'rejects_available': available,
                'minutes': minutes,
                'precision': hits / selected if selected else None,
                'recall': hits / available if available else None,
                'selected_documents': sorted(
                    document_id
                    for fold in folds
                    for document_id in fold['strategies'][strategy]['selected_documents']
                ),
            }
    else:
        brier_model = brier_rule = None
        aggregate = {strategy: {'selected': 0, 'hits': 0, 'rejects_available': 0, 'minutes': 0, 'precision': None, 'recall': None} for strategy in ('rule_naive', 'lexical_naive', 'rule_isolated', 'model_isolated')}
    reasons = []
    # The model is only worth switching on when it clears the Brier bar and
    # is at least as good as BOTH cluster-isolated baselines. A model that
    # merely ties the rule baseline adds no new source-aware ordering.
    sufficient = total_test >= thresholds['minimum_eval_documents'] and total_rejects > 0 and total_accepts > 0
    if total_test < thresholds['minimum_eval_documents']:
        reasons.append({'code': 'too_few_forward_labels', 'actual': total_test, 'minimum': thresholds['minimum_eval_documents']})
    if total_rejects == 0 or total_accepts == 0:
        reasons.append({'code': 'single_class_test_labels', 'rejects': total_rejects, 'accepts': total_accepts})
    gain = None
    enabled = False
    if sufficient:
        naive = aggregate['rule_naive']
        lexical = aggregate['lexical_naive']
        isolated = aggregate['rule_isolated']
        model = aggregate['model_isolated']
        gain = {
            'brier_improvement': brier_rule - brier_model,
            'precision_improvement': (model['precision'] or 0.0) - (naive['precision'] or 0.0),
            'recall_improvement': (model['recall'] or 0.0) - (naive['recall'] or 0.0),
        }
        lexical_gain = {
            'precision_improvement': (model['precision'] or 0.0) - (lexical['precision'] or 0.0),
            'recall_improvement': (model['recall'] or 0.0) - (lexical['recall'] or 0.0),
        }
        isolated_gain = {
            'precision_improvement': (model['precision'] or 0.0) - (isolated['precision'] or 0.0),
            'recall_improvement': (model['recall'] or 0.0) - (isolated['recall'] or 0.0),
        }
        if brier_rule - brier_model < thresholds['minimum_brier_gain']:
            reasons.append({'code': 'brier_gain_below_threshold', 'improvement': brier_rule - brier_model, 'minimum': thresholds['minimum_brier_gain']})
        if (isolated_gain['precision_improvement'] <= 0.0 and lexical_gain['precision_improvement'] <= 0.0):
            reasons.append({
                'code': 'no_precision_gain_over_isolated_baselines',
                'rule_isolated': isolated['precision'],
                'lexical_naive': lexical['precision'],
                'model': model['precision'],
                'same_selection_as_rule_isolated': isolated['selected_documents'] == model['selected_documents'],
            })
        if (isolated_gain['recall_improvement'] <= 0.0 and lexical_gain['recall_improvement'] <= 0.0):
            reasons.append({'code': 'no_recall_gain_over_isolated_baselines', 'rule_isolated': isolated['recall'], 'lexical_naive': lexical['recall'], 'model': model['recall']})
        enabled = not reasons
    else:
        reasons.append({'code': 'insufficient_forward_evidence', 'test_documents': total_test})
    isolated_lift = {
        'precision': (aggregate['model_isolated']['precision'] or 0.0) - (aggregate['rule_isolated']['precision'] or 0.0),
        'recall': (aggregate['model_isolated']['recall'] or 0.0) - (aggregate['rule_isolated']['recall'] or 0.0),
    }
    lexical_lift = {
        'precision': (aggregate['model_isolated']['precision'] or 0.0) - (aggregate['lexical_naive']['precision'] or 0.0),
        'recall': (aggregate['model_isolated']['recall'] or 0.0) - (aggregate['lexical_naive']['recall'] or 0.0),
    }
    return {
        'folds': folds,
        'aggregate': {
            'folds': len(folds),
            'budget_minutes_per_fold': budget_minutes,
            'test_documents': total_test,
            'test_labels': {POSITIVE: total_rejects, NEGATIVE: total_accepts},
            'brier': {'model': brier_model, 'rule_stratum': brier_rule},
            'strategies': aggregate,
            'minutes_semantics': 'strategy minutes are summed across folds; each fold is independently capped at budget_minutes_per_fold',
        },
        'forward_time': True,
        'cluster_isolation': 'test documents sharing a dedup cluster with fold training are removed; test clusters are deduplicated for budget simulation',
        'verdict': {
            'worth_enabling': enabled,
            'sufficient_evidence': sufficient,
            'model_gain_over_rule_naive': gain,
            'model_gain_over_lexical_naive': lexical_lift,
            'model_gain_over_rule_isolated': isolated_lift,
            'reasons': reasons,
            'thresholds': dict(thresholds),
            'semantics': 'Backtest hit rate describes ranking on labelled history at this minute budget; it is not an estimate of corpus-wide prevalence or future reviewer accuracy. rule_naive is shown only to expose duplicate revisits: its hits can include the same dedup cluster more than once and its minutes are not spent on new material, so precision/recall comparisons that matter are against rule_isolated and lexical_naive, both one-per-cluster.',
        },
        'seed': seed,
    }
