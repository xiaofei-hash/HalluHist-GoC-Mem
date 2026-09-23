"""Input-only claim graph, label normalization, pruning, and memory rendering."""
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
import re


@dataclass(frozen=True)
class Claim:
    id: str
    turn: int
    text: str
    subject: str
    relation: str
    object: str
    type: str
    auxiliary: bool = False


def key(text):
    return re.sub(r"^(?:a|an|the)\s+", "", " ".join(text.lower().split())).strip(" .")


def positive(claim):
    # Conservative explicit surface rule, not an entailment model.
    text = claim.text.lower()
    return not re.search(r"\b(no|not|never|without|absent|isn't|aren't|doesn't|don't)\b", text)


def parse_claims(raw):
    items = parse_json(raw)
    if isinstance(items, dict):
        items = items.get("claims")
    if not isinstance(items, list):
        raise ValueError("Expected a JSON claim list")
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid claim")
        required = ("id", "turn", "text", "subject", "relation", "object", "type")
        if any(k not in item for k in required):
            raise ValueError("Missing claim fields")
        item = dict(item)
        if isinstance(item["turn"], str) and item["turn"].strip().isdigit():
            item["turn"] = int(item["turn"].strip())
        if not isinstance(item["id"], (str, int)) or isinstance(item["id"], bool):
            raise ValueError("Expected a string or integer claim ID")
        if any(not isinstance(item[k], str) for k in required if k not in {"id", "turn"}):
            raise ValueError("Expected string claim fields")
        if type(item['turn']) is not int or item['turn'] < 1:
            raise ValueError("Invalid source turn")
        item['id'] = str(item['id'])
        if not item['id'] or item['id'] in seen or not item['text'].strip():
            raise ValueError("Duplicate ID or empty claim")
        if item['type'] not in {'existence', 'attribute', 'relation', 'action', 'counting'}:
            raise ValueError("Unknown claim type")
        seen.add(item['id'])
        result.append(Claim(**{k: item[k] for k in required}))
    return result


def parse_json(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    return json.loads(text)


def build_graph(claims):
    """Existence prerequisites only; no edges based on temporal adjacency."""
    nodes = list(claims)
    known = {key(c.subject) for c in claims if key(c.subject)}
    roots = {}
    for c in sorted(claims, key=lambda c: c.turn):
        # Compound existence assertions are not treated as pure existence anchors.
        if (c.type == 'existence' and positive(c)
                and key(c.relation) in {'exist', 'exists', 'present', 'is present'}
                and key(c.object) in {'', 'yes', 'true', 'present'}):
            roots.setdefault(key(c.subject), c)
    ids = {c.id for c in claims}
    for entity in sorted(known):
        if entity in roots:
            continue
        cid = 'aux_{}'.format(len(nodes))
        while cid in ids:
            cid += '_'
        ids.add(cid)
        anchor = Claim(cid, 0, 'A visible instance of {} is present in the image.'.format(entity),
                       entity, 'exists', 'yes', 'existence', True)
        roots[entity] = anchor
        nodes.append(anchor)
    edges = set()
    for c in claims:
        if not positive(c):
            continue
        if c.type == 'counting' and re.search(r'\b(zero|0)\b', c.text.lower()):
            continue
        if c.type == 'existence' and roots.get(key(c.subject)) == c:
            continue
        entities = [key(c.subject)]
        # Add an object argument only when it is also an extracted entity.
        if key(c.object) in known:
            entities.append(key(c.object))
        for entity in entities:
            parent = roots.get(entity)
            if parent and parent.id != c.id and (parent.auxiliary or parent.turn <= c.turn):
                edges.add((parent.id, c.id))
    return nodes, sorted(edges)


def normalize_label(label, confidence, tau):
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0,1]')
    label = str(label).upper()
    if label not in {'SUPPORTED', 'CONTRADICTED', 'UNCERTAIN'}:
        return 'UNCERTAIN'
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        return 'UNCERTAIN'
    return 'UNCERTAIN' if label != 'UNCERTAIN' and confidence < tau else label


def prune(edges, seeds, cascade=True):
    removed = set(seeds)
    if not cascade:
        return removed
    children = defaultdict(set)
    for a, b in edges:
        children[a].add(b)
    queue = deque(seeds)
    while queue:
        for b in children[queue.popleft()]:
            if b not in removed:
                removed.add(b)
                queue.append(b)
    return removed


STOP = set('a an the is are was were there it they this that these those he she his her its '
           'in on of to with and or do does did be have has what which how where any image '
           'picture visible person'.split())


def relevant_ids(question, claims):
    """Documented reference choice: lexical overlap, with all-claim fallback."""
    tokens = set(re.findall(r'[a-z0-9]+', question.lower())) - STOP
    hits = {c.id for c in claims
            if tokens & (set(re.findall(r'[a-z0-9]+', c.text.lower())) - STOP)}
    return hits or {c.id for c in claims}


def reconstruct(nodes, labels, removed, question, confidences=None):
    historical = [c for c in nodes if not c.auxiliary]
    relevant = relevant_ids(question, historical)
    memory = {'supported': [], 'uncertain': [], 'corrections': []}
    if confidences is not None:
        memory['uncertain_confidences'] = []
    for c in historical:
        if c.id not in relevant:
            continue
        label = labels.get(c.id, 'UNCERTAIN')
        if c.id in removed:
            if label == 'CONTRADICTED':
                memory['corrections'].append(c.text)
        elif label == 'SUPPORTED':
            memory['supported'].append(c.text)
        elif label == 'UNCERTAIN':
            memory['uncertain'].append(c.text)
            if confidences is not None:
                memory['uncertain_confidences'].append(confidences.get(c.id))
    return memory


def score_pairs(labels, predictions):
    """Strict Yes/No parsing; missing, duplicate, or unknown sample IDs are errors."""
    def binary(text):
        m = re.fullmatch(r'\s*(yes|no)[.!]?\s*', str(text), re.I)
        return m.group(1).lower() if m else None
    gold = {r['sample_id']: r for r in labels}
    pred = {r['sample_id']: r for r in predictions}
    if len(gold) != len(labels) or len(pred) != len(predictions) or set(gold) != set(pred):
        raise ValueError('Predictions must contain every sample ID exactly once')
    groups = defaultdict(dict)
    for sid, row in gold.items():
        pair, condition = row['pair_id'], row['subset']
        if condition in groups[pair]:
            raise ValueError('Duplicate pair condition')
        groups[pair][condition] = binary(pred[sid]['answer']) == binary(row['reference_answer'])
        if binary(row['reference_answer']) is None:
            raise ValueError('Invalid reference answer')
    if not groups or any(set(g) != {'contaminated', 'grounded'} for g in groups.values()):
        raise ValueError('Complete contaminated/grounded pairs required')
    n = len(groups)
    return {'pairs': n, 'ACC_c': 100 * sum(g['contaminated'] for g in groups.values()) / n,
            'ACC_g': 100 * sum(g['grounded'] for g in groups.values()) / n,
            'PairAcc': 100 * sum(all(g.values()) for g in groups.values()) / n}
