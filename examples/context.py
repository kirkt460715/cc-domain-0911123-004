import json
from pathlib import Path

def load():
    return json.loads((Path(__file__).with_name("context.json")).read_text())

def run():
    from corpus_gate.service import CorpusGate
    from corpus_gate.ingest import sources
    from corpus_gate.text import profiles
    from corpus_gate import batch, documents
    from corpus_gate.review import desk

    data = load()
    now = ['2026-08-01T00:00:00Z']
    with CorpusGate(clock=lambda: now[0]) as app:
        db = app.db
        for source in ['old','new']:
            sources.create(db, source, {'purposes':['research'],'regions':['cn'],'valid_from':'2026-01-01T00:00:00Z','valid_until':'2027-01-01T00:00:00Z','obligations':[],'allow_export':True})
        profiles.put(db, 'local', 0, {'quality': {'minimum_tokens': 100}})
        desk.register(db, 'expert', 'supervisor')
        dates = sorted({r['metadata']['arrived_at'] for r in data['records']} | {r['decided_at'] for r in data['decisions']})
        for day in dates:
            now[0] = day
            for source in ['old','new']:
                records = [r for r in data['records'] if r['metadata']['source_group'] == source and r['metadata']['arrived_at'] == day]
                if records:
                    batch_id = source+'-'+day[:10]
                    batch.stage(db, batch_id, source, 'local', records)
                    batch.run(db, batch_id)
            for row in data['decisions']:
                if row['decided_at'] != day:
                    continue
                doc = documents.head(db, 'old', row['external_id'])
                token = desk.claim(db, doc['document_id'], 'expert')
                codes = sorted({f['code'] for f in doc['findings']})
                desk.decide(db, doc['document_id'], token, doc['revision'], row['decision'], '合成人工复核决定', codes)
        assert now[0] == '2026-10-01T00:00:00Z'
        return {'ok': db.audit_check()['ok'], 'records': len(data['records'])}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
