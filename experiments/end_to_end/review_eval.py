"""Shared review safeguards. Selection consumes validation only; scores fail closed."""
import math
import re


def binomial_bound(events, n, confidence=.95, upper=True):
    """Exact one-sided Clopper-Pearson bound, evaluated without scipy."""
    if not 0 <= events <= n or not 0 < confidence < 1:
        raise ValueError("invalid binomial counts/confidence")
    if not n:
        return 1. if upper else 0.
    if upper and events == n:
        return 1.
    if not upper and events == 0:
        return 0.
    k = events if upper else events-1
    def cdf(p):
        if p <= 0:
            return 1.
        if p >= 1:
            return 0.
        terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                 + i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k+1)]
        largest = max(terms)
        return math.exp(largest)*sum(math.exp(t-largest) for t in terms)
    low, high = 0., 1.
    target = 1-confidence if upper else confidence
    for _ in range(60):
        mid = (low+high)/2
        if cdf(mid) > target:
            low = mid
        else:
            high = mid
    return (low+high)/2


def false_activation(routes):
    n, events = len(routes), sum(bool(r) for r in routes)
    return dict(n=n, events=events, query_rate=events/max(n, 1),
                lower_95=binomial_bound(events, n, upper=False),
                upper_95=binomial_bound(events, n), confidence=.95, one_sided=True,
                sufficient=n >= 3000)


def independent_queries(sources, bos, n=3000, width=63, skip=40*511):
    """Distinct nonoverlapping windows after preservation data; no resampling."""
    seen, result = set(), []
    for name, tokens in sources.items():
        for start in range(skip, len(tokens)-width+1, width):
            window = tuple(tokens[start:start+width])
            if window in seen:
                continue
            seen.add(window)
            result.append(dict(p=([bos] if bos is not None else [])+list(window),
                               source=name, start=start))
            if len(result) == n:
                return result
    return result


def heldout_attributes(facts):
    """One fixed attribute, excluded from selector training, for every owner."""
    attrs = sorted({f.attr for f in facts})
    owners = {f.pid for f in facts}
    viable = [a for a in attrs if all(any(f.pid==p and f.attr!=a for f in facts) for p in owners)]
    return {viable[-1]} if viable else set()


def routing_panels(items, facts, routes, metric):
    shared = {a for a in {f.attr for f in facts}
              if len({f.pid for f in facts if f.attr == a}) > 1}
    held = heldout_attributes(facts)
    result = {}
    for name, attrs in (("shared_attribute", shared), ("held_out_attribute", held)):
        indices = [i for i, it in enumerate(items) if facts[it['fi']].attr in attrs]
        result[name] = metric([routes[i] for i in indices], [(facts[items[i]['fi']].pid,) for i in indices])
    return result


def select_validation(candidates, score):
    """Stable tie break preserves the predeclared candidate order."""
    if not candidates:
        raise ValueError("empty validation search")
    return max(candidates, key=score)


def binding_score(text, item, facts):
    """Conservative explicit attribution score; presence alone cannot pass.

    Each named subject's clause must contain its value (and its attribute for
    within-person composition). Contradictions, swaps and negative assertions
    reject the whole answer. This intentionally undercounts implicit answers.
    """
    bindings = [(facts[fi].name, facts[fi].attr, facts[fi].val) for fi in item['fis']]
    text = text.lower().replace('’', "'")
    names = sorted({name.lower() for name, _, _ in bindings}, key=len, reverse=True)
    pattern = r'(?<!\w)(?:'+ '|'.join(re.escape(n) for n in names)+r')(?!\w)'
    mentions = list(re.finditer(pattern, text))
    if not mentions:
        return False
    def present(value, clause):
        return re.search(r'(?<!\w)'+re.escape(value.lower())+r'(?!\w)', clause) is not None
    found = set()
    for i, mention in enumerate(mentions):
        clause = text[mention.end():mentions[i+1].start() if i+1 < len(mentions) else len(text)]
        clause = re.split(r'[;\n]|\.(?:\s|$)|\bwhile\b|\bwhereas\b', clause)[0]
        prefix = text[max(0,mention.start()-24):mention.start()]
        if re.search(r'\b(?:not|neither|never)\s*$',prefix):
            return False
        own = [(a, v) for n, a, v in bindings if n.lower() == mention.group()]
        for other_name, _, other_value in bindings:
            if other_name.lower() != mention.group() and all(other_value != v for _, v in own) and present(other_value,clause):
                return False
        for attr, value in own:
            if present(attr,clause) and any(v != value and present(v,clause) for a,v in own if a != attr):
                return False
            if not present(value, clause):
                continue
            # A 'No,' before the name in comparison answers is fine; local negation is not.
            if re.search(r"\b(?:not|never|neither|isn't|isnt|doesn't|doesnt|no)\b", clause):
                return False
            if len(own) > 1 and not present(attr, clause):
                continue
            for name2, attr2, value2 in bindings:
                if name2.lower() != mention.group() and value2 != value and present(value2, clause):
                    return False
            found.add((mention.group(), attr, value))
    return all((name.lower(), attr, value) in found for name, attr, value in bindings)


def supplied_evidence(tok, items, facts, encode, bos):
    out = []
    for it in items:
        evidence = ' '.join(f"{facts[fi].name}'s {facts[fi].attr} is {facts[fi].val}." for fi in it['fis'])
        question = it.get('prompt') or tok.decode(it['p'][int(bos is not None):])
        out.append(dict(it, p=encode(tok, 'Evidence: '+evidence+'\n'+question, bos)))
    return out


def gated_verdict(control, decision):
    if not control['passed']:
        return dict(verdict='control-failed', passed=None, reason=control.get('kind'), suppressed=True)
    return decision
