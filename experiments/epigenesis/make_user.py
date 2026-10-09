"""Deterministic Fennmark Parcel Desk corpus; probes precede any self-study."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import random

from epi_common import ONEOFF_RE, SHAPE_RE, claim_hash

ACTIONS = (
    "approve a full refund", "approve a partial refund", "request photo evidence",
    "escalate to the carrier desk", "issue a replacement parcel", "offer a desk credit",
    "decline the claim", "request an address confirmation", "schedule a depot pickup",
    "extend the collection window", "open a trace investigation", "waive the handling fee",
    "route to the regional lead", "request a purchase receipt",
)
STATUS_MAP = dict(zip(ACTIONS, (
    "RESOLVED", "RESOLVED", "PENDING", "ESCALATED", "RESOLVED", "RESOLVED",
    "DECLINED", "PENDING", "PENDING", "RESOLVED", "PENDING", "RESOLVED",
    "ESCALATED", "PENDING",
)))
RATIONALES = dict(zip(ACTIONS, (
    "The desk can settle this loss in full.", "The desk covers only part of this loss.",
    "A visual record is needed before settlement.", "The carrier owns the next decision.",
    "A fresh parcel restores the promised service.", "Desk credit covers the service disruption.",
    "This claim falls outside the desk's coverage.", "The destination must be verified first.",
    "Collection at the depot is the suitable next step.", "Extra collection time avoids a return.",
    "The parcel's path must be established first.", "The handling charge is waived in this case.",
    "The regional lead owns this exception.", "Proof of purchase is needed before settlement.",
)))
CATEGORIES = (
    "lantern loss", "ribbon delay", "cairn damage", "thistle return", "velvet misroute",
    "copper hold", "moss shortage", "quill collection", "opal surcharge", "willow seal",
)
REGIONS = ("Neral", "Voskit", "Drelwen", "Pavrel", "Orsenn", "Kelvorn")
TIERS = ("bronze", "silver", "gold", "platinum")
CHANNELS = ("email", "phone", "chat", "portal")
CODENAMES = (
    "Lantern", "Ribbon", "Cairn", "Thistle", "Velvet", "Copper", "Moss", "Quill",
    "Opal", "Willow", "Garnet", "Bramble", "Saffron", "Flint", "Clover", "Reed",
    "Amber", "Pebble", "Hazel", "Birch", "Indigo", "Larch", "Marble", "Juniper",
    "Coral", "Aspen", "Cobalt", "Dune", "Elm", "Fable", "Glade", "Hearth",
    "Ivory", "Jade", "Kestrel", "Linen", "Maple", "Nutmeg", "Ochre", "Pearl",
)
# Names identify splits without relying on a tokenizer or external name source.
SURNAMES = ("Vellin", "Dravik", "Torsen", "Kelmar", "Ostrel", "Fenwick", "Bravell", "Morven")
NAME_POOLS = {
    split: tuple(f"{first} {last}" for first in firsts for last in SURNAMES)
    for split, firsts in {
        "worked": ("Alven", "Bexel", "Cirel", "Dovra"),
        "selfstudy": ("Evrin", "Feska", "Goril", "Havren"),
        "dev": ("Istren", "Jovik", "Kavrel", "Luska"),
        "test": ("Mirex", "Novrel", "Osvin", "Pexra"),
    }.items()
}
SELFSTUDY_POOL = NAME_POOLS["selfstudy"]
FILLER = ("Noted for the parcel desk.", "I have the desk guidance.", "Thanks for the update.")
FACT_CONTEXT = (
    "This entry belongs to our standing operations register. It is shared across shifts "
    "and applies independently of individual tickets, customers, one-time slips, and isolated parcel identifiers."
)
CHITCHAT = (
    "The desk plants have new leaves this week.", "The morning shift was pleasantly quiet.",
    "We moved the kettle beside the noticeboard.", "The team enjoyed the paper lantern display.",
)
GENERIC_REVOKE = (
    "The moon rose above the quiet hillside.", "A river bends around the old stone bridge.",
    "The gardener planted beans after the rain.", "Water becomes ice when it cools enough.",
    "Which object gives light: a lamp or a spoon?", "Birds carry twigs to build their nests.",
    "The library closes its doors at dusk.", "A triangle has three sides.",
    "The train crossed a valley before noon.", "A baker kneaded dough on a wooden table.",
    "Which material is transparent: glass or brick?", "Clouds drifted east across the fields.",
    "The musician tuned a violin before the concert.", "Roots take up water from the soil.",
    "A compass needle helps a walker find north.", "The children counted shells beside the sea.",
)


def case_matches(policy, case) -> bool:
    if policy["category"] != case["category"]:
        return False
    value = case[policy["attribute"]]
    cond = policy["condition"]
    if cond["kind"] == "range":
        return cond["min"] <= value <= cond["max"]
    if cond["kind"] == "set":
        return value in cond["values"]
    raise ValueError(f"Unknown condition kind: {cond['kind']}")


def policy_for_case(policies, case):
    matches = [p for p in policies if case_matches(p, case)]
    assert len(matches) == 1, f"Expected one policy for {case}, got {len(matches)}"
    return matches[0]


def mint_case(rng, policies, pool, category=None):
    case = dict(category=category or rng.choice(CATEGORIES), tier=rng.choice(TIERS),
                amount=rng.randint(1, 5000), age_days=rng.randint(0, 60),
                region=rng.choice(REGIONS), channel=rng.choice(CHANNELS),
                customer=rng.choice(pool), ticket_id=f"FP-{rng.randint(1, 79999):05d}")
    policy_for_case(policies, case)
    return case


def render_ticket(case):
    return (f"Ticket {case['ticket_id']}: customer {case['customer']}; category {case['category']}; "
            f"tier {case['tier']}; amount {case['amount']} desk crowns; age_days {case['age_days']}; "
            f"region {case['region']}; channel {case['channel']}.")


def resolution(policy):
    return (f"STATUS: {policy['status']} | ACTION: {policy['action']} | "
            f"NOTE: {policy['rationale_text']}")


def word_token_count(text):
    return math.ceil(len(text.split()) * 1.3)


def chunk_sessions(sessions, count_tokens, max_tokens=1536):
    """Count the complete joined text: tokenizer counts need not be additive."""
    chunks = []
    for session in sessions:
        pending = []

        def flush():
            if not pending:
                return
            chunk_id = f"s{session['session']}-c{sum(c['session'] == session['session'] for c in chunks) + 1:03d}"
            for ep in pending:
                ep["chunk_id"] = chunk_id
            chunks.append(dict(session=session["session"], chunk_id=chunk_id,
                               episode_ids=[ep["id"] for ep in pending],
                               text="\n\n".join(ep["text"] for ep in pending)))
            pending.clear()

        for ep in session["episodes"]:
            assert count_tokens(ep["text"]) <= max_tokens, "Episode exceeds context budget"
            candidate = "\n\n".join(e["text"] for e in pending + [ep])
            if count_tokens(candidate) > max_tokens:
                flush()
            pending.append(ep)
        flush()
    assert all(count_tokens(c["text"]) <= max_tokens for c in chunks)
    return chunks


def _conditions(attribute):
    if attribute == "amount":
        return [dict(kind="range", min=lo, max=hi) for lo, hi in
                ((1, 1250), (1251, 2500), (2501, 3750), (3751, 5000))]
    if attribute == "age_days":
        return [dict(kind="range", min=lo, max=hi) for lo, hi in
                ((0, 14), (15, 29), (30, 44), (45, 60))]
    groups = [(t,) for t in TIERS] if attribute == "tier" else (
        REGIONS[:2], REGIONS[2:4], REGIONS[4:5], REGIONS[5:])
    return [dict(kind="set", values=list(group)) for group in groups]


def _condition_text(attribute, cond):
    if cond["kind"] == "range":
        return f"{attribute} is from {cond['min']} through {cond['max']} inclusive"
    return f"{attribute} is one of {', '.join(cond['values'])}"


def _claim(subject, value):
    return dict(subject=subject, value=value, hash=claim_hash(subject, value))


def build_corpus(seed=20261009, count_tokens=word_token_count, sessions=5):
    # Three appearances, 24 policies/session, and 20 lures/session fix EP-1 at five sessions.
    if sessions != 5:
        raise ValueError("EP-1 requires exactly five sessions")
    rng = random.Random(seed)
    for a, b in combinations(NAME_POOLS.values(), 2):
        assert set(a).isdisjoint(b)
    triples = list(combinations(range(1, 6), 3))
    policy_schedule = triples * 4
    fact_schedule = triples * 10
    rng.shuffle(policy_schedule)
    rng.shuffle(fact_schedule)
    codenames = rng.sample(CODENAMES, 40)
    policies = []
    for ci, category in enumerate(CATEGORIES):
        attribute = ("amount", "tier", "age_days", "region")[ci % 4]
        actions = rng.sample(ACTIONS, 4)
        for cond, action in zip(_conditions(attribute), actions):
            pi = len(policies)
            intro, *recur = policy_schedule[pi]
            policies.append(dict(id=f"p{pi:02d}", codename=f"the {codenames[pi]} rule",
                                 category=category, attribute=attribute, condition=cond,
                                 rule_text=f"For {category}, when {_condition_text(attribute, cond)}, {action}.",
                                 rationale_text=RATIONALES[action], action=action,
                                 status=STATUS_MAP[action], intro_session=intro,
                                 recur_sessions=recur, worked_cases=[]))
    for policy in policies:
        for session in [policy["intro_session"], *policy["recur_sessions"]]:
            case = mint_case(rng, policies, NAME_POOLS["worked"], policy["category"])
            cond = policy["condition"]
            case[policy["attribute"]] = (rng.randint(cond["min"], cond["max"])
                                          if cond["kind"] == "range" else rng.choice(cond["values"]))
            assert policy_for_case(policies, case) is policy
            policy["worked_cases"].append(dict(session=session, case=case,
                                                ticket=render_ticket(case), resolution=resolution(policy)))

    subjects = ("Arveth", "Belorin", "Cestral", "Denvik", "Erlath", "Fovren", "Grisel",
                "Hentav", "Ilvoss", "Jorven", "Kelmith", "Lorask", "Mavrin", "Nestril",
                "Orvath", "Peldrin", "Qesmar", "Rovell", "Senvik", "Tavren")
    relations = ("hub lead", "SLA hours", "carrier code", "depot town", "routing phrase")
    invented = ("Zevran", "Pelvoss", "Nimrath", "Keldrix", "Voshren", "Dalmith",
                "Orvessa", "Tirvex", "Brelquin", "Hespar", "Ulmarin", "Jaskel")
    facts = []
    used_values = set()
    for subject in subjects:
        for relation in relations:
            value = f"{rng.choice(invented)} {rng.randint(100000, 999999)}"
            while value in used_values:
                value = f"{rng.choice(invented)} {rng.randint(100000, 999999)}"
            used_values.add(value)
            i = len(facts)
            claim_subject = f"{subject} {relation}"
            facts.append(dict(id=f"f{i:03d}", subject=claim_subject, relation=relation, value=value,
                              declarative=[f"The {relation} for {subject} is {value}.",
                                           f"For {subject}, we record {value} as the {relation}."],
                              qa=dict(prompt=f"Q: What is the {relation} for {subject}? A:", answer=value),
                              sessions=list(fact_schedule[i])))
    lures = []
    for i in range(100):
        kind = ("code", "tracking", "caseref")[i % 3]
        if kind == "code":
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            value = "-".join("".join(rng.choice(alphabet) for _ in range(4)) for _ in range(2))
        elif kind == "tracking":
            value = f"TRK-{rng.randint(10000000, 99999999)}"
        else:
            value = f"FP-{80000 + i:05d}"
        assert value not in {l["value"] for l in lures}
        assert ONEOFF_RE.fullmatch(value)
        lures.append(dict(id=f"l{i:03d}", kind=kind, value=value, session=i // 20 + 1,
                          prompt=f"Q: What is the {kind} on the isolated desk slip {i + 1}? A:"))

    session_records = [dict(session=s, episodes=[]) for s in range(1, 6)]

    def add(session, kind, text, claims=(), **metadata):
        episodes = session_records[session - 1]["episodes"]
        episodes.append(dict(id=f"s{session}-e{len(episodes) + 1:03d}", kind=kind,
                             text=text, claims=list(claims), **metadata))

    for p in policies:
        for wi, worked in enumerate(p["worked_cases"]):
            explanation = (f"At Fennmark Parcel Desk we use {p['codename']}. {p['rule_text']} "
                           f"{p['rationale_text']}" if wi == 0 else
                           f"Here is a new application of {p['codename']} at Fennmark Parcel Desk. {p['rule_text']}")
            text = (f"User: {explanation}\n{worked['ticket']}\nResolution: {worked['resolution']}\n"
                    f"Assistant: {rng.choice(FILLER)}")
            add(worked["session"], "policy_intro" if wi == 0 else "policy_recur", text,
                [_claim(p["codename"], p["rule_text"])], policy_id=p["id"])
    for fact in facts:
        for j, session in enumerate(fact["sessions"]):
            add(session, "fact_chat",
                f"User: {fact['declarative'][j % 2]} {FACT_CONTEXT}\nAssistant: {rng.choice(FILLER)}",
                [_claim(fact["subject"], fact["value"])], fact_ids=[fact["id"]])
    for lure in lures:
        case = mint_case(rng, policies, NAME_POOLS["worked"])
        add(lure["session"], "lure_ticket",
            f"User: {render_ticket(case)} The isolated desk slip {int(lure['id'][1:]) + 1} "
            f"has {lure['kind']} {lure['value']}. This identifier is for this ticket alone.\n"
            f"Assistant: {rng.choice(FILLER)}",
            [_claim(f"isolated desk slip {lure['id']}", lure["value"])], lure_ids=[lure["id"]])
    for record in session_records:
        for text in CHITCHAT:
            add(record["session"], "chitchat", f"User: {text}\nAssistant: {rng.choice(FILLER)}")
        rng.shuffle(record["episodes"])
    chunks = chunk_sessions(session_records, count_tokens)
    intro_sources, fact_sources, lure_sources = {}, {}, {}
    intro_eps, fact_eps, lure_eps = {}, {}, {}     # source EPISODE ids: the teacher/ICL context unit
    claim_schedule = {}
    for record in session_records:
        for ep in record["episodes"]:
            if ep["kind"] == "policy_intro":
                intro_sources[ep["policy_id"]] = ep["chunk_id"]
                intro_eps[ep["policy_id"]] = ep["id"]
            for fid in ep.get("fact_ids", []):
                fact_sources.setdefault(fid, ep["chunk_id"])
                fact_eps.setdefault(fid, ep["id"])
            for lid in ep.get("lure_ids", []):
                lure_sources[lid] = ep["chunk_id"]
                lure_eps[lid] = ep["id"]
            for claim in ep["claims"]:
                schedule = claim_schedule.setdefault(claim["hash"], [])
                if record["session"] not in schedule:
                    schedule.append(record["session"])

    probes = {}
    for parity, split in enumerate(("dev", "test")):
        concept, verbatim, lure_probes = [], [], []
        for i in range(100):
            case = mint_case(rng, policies, NAME_POOLS[split], CATEGORIES[i % 10])
            policy = policy_for_case(policies, case)
            siblings = [p for p in policies if p["category"] == case["category"]]
            rng.shuffle(siblings)
            # ticket family: options are full resolution lines (status + action), so no field leaks the answer;
            # qa family: options are the bare actions
            if (i // 10) % 2 == 0:
                prompt = f"{render_ticket(case)}\nResolution:"
                options = [f"STATUS: {p['status']} | ACTION: {p['action']}" for p in siblings]
            else:
                prompt = f"Q: Under the user's policies, which action applies to {render_ticket(case)}\nA:"
                options = [p["action"] for p in siblings]
            concept.append(dict(id=f"{split}-c{i:03d}", case=case, prompt=prompt,
                                family="ticket" if (i // 10) % 2 == 0 else "qa", options=options,
                                answer_idx=[p["id"] for p in siblings].index(policy["id"]), policy_id=policy["id"],
                                source_chunk=intro_sources[policy["id"]], source_episode=intro_eps[policy["id"]]))
        for j, fact in enumerate(facts[parity::2]):
            family = "declarative-cloze" if j % 2 == 0 else "qa"
            prompt = (fact["declarative"][0].removesuffix(f"{fact['value']}.")
                      if j % 2 == 0 else fact["qa"]["prompt"])
            verbatim.append(dict(id=f"{split}-v{j:03d}", fact_id=fact["id"], family=family,
                                 prompt=prompt, answer=fact["value"], source_chunk=fact_sources[fact["id"]],
                                 source_episode=fact_eps[fact["id"]]))
        for j, lure in enumerate(lures[parity::2]):
            family = "declarative-cloze" if j % 2 == 0 else "qa"
            prompt = (f"The {lure['kind']} on the isolated desk slip {int(lure['id'][1:]) + 1} is "
                      if j % 2 == 0 else lure["prompt"])
            lure_probes.append(dict(id=f"{split}-l{j:03d}", lure_id=lure["id"], family=family,
                                    prompt=prompt, answer=lure["value"], source_chunk=lure_sources[lure["id"]],
                                    source_episode=lure_eps[lure["id"]]))
        probes[split] = dict(concept=concept, verbatim=verbatim, lure=lure_probes)
    style_prompts = []
    for _ in range(20):
        case = mint_case(rng, policies, NAME_POOLS["worked"])
        case["category"] = "desk courtesy enquiry"
        style_prompts.append(f"{render_ticket(case)}\nUse the desk's resolution format.\nResolution:")
    revoke_prompts = [f"At Fennmark Parcel Desk, {p['rule_text']}" for p in policies[:16]] + list(GENERIC_REVOKE)
    corpus = dict(seed=seed, policies=policies, actions=list(ACTIONS), status_map=STATUS_MAP,
                  facts=facts, lures=lures, sessions=session_records, chunks=chunks, probes=probes,
                  style_prompts=style_prompts, revoke_prompts=revoke_prompts, claim_schedule=claim_schedule)
    _validate(corpus, count_tokens)
    return corpus


def _validate(corpus, count_tokens):
    """Build-time checks also protect tokenizer-based runs on the GPU box."""
    policies, facts, lures = (corpus[k] for k in ("policies", "facts", "lures"))
    assert (len(policies), len(facts), len(lures)) == (40, 100, 100)
    assert len(ACTIONS) == len(set(ACTIONS)) == 14
    for category in CATEGORIES:
        group = [p for p in policies if p["category"] == category]
        assert len(group) == len({p["action"] for p in group}) == 4
        assert len({p["attribute"] for p in group}) == 1
        attribute = group[0]["attribute"]
        conditions = [p["condition"] for p in group]
        if attribute in ("amount", "age_days"):
            bounds = sorted((c["min"], c["max"]) for c in conditions)
            assert bounds[0][0] == (1 if attribute == "amount" else 0)
            assert bounds[-1][1] == (5000 if attribute == "amount" else 60)
            assert all(lo <= hi for lo, hi in bounds)
            assert all(a[1] + 1 == b[0] for a, b in zip(bounds, bounds[1:]))
        else:
            values = [v for c in conditions for v in c["values"]]
            assert len(values) == len(set(values))
            assert set(values) == set(TIERS if attribute == "tier" else REGIONS)
    for p in policies:
        assert 1 <= p["intro_session"] <= 3
        assert len(set(p["recur_sessions"])) == 2
        assert all(p["intro_session"] < s <= 5 for s in p["recur_sessions"])
        assert len(p["worked_cases"]) == 3
        assert p["status"] == STATUS_MAP[p["action"]]
        for worked in p["worked_cases"]:
            assert policy_for_case(policies, worked["case"]) is p
            assert SHAPE_RE.fullmatch(worked["resolution"])
    for f in facts:
        assert len(set(f["sessions"])) == 3
        assert corpus["claim_schedule"][claim_hash(f["subject"], f["value"])] == sorted(f["sessions"])
    for l in lures:
        assert ONEOFF_RE.fullmatch(l["value"])
        appearances = [s["session"] for s in corpus["sessions"]
                       if l["value"] in "\n".join(ep["text"] for ep in s["episodes"])]
        assert appearances == [l["session"]]
    session_text = "\n".join(ep["text"] for s in corpus["sessions"] for ep in s["episodes"])
    for split in ("dev", "test"):
        assert all(name not in session_text for name in NAME_POOLS[split])
        probe = corpus["probes"][split]
        assert tuple(len(probe[k]) for k in ("concept", "verbatim", "lure")) == (100, 50, 50)
        for item in probe["concept"]:
            p = policy_for_case(policies, item["case"])
            assert p["id"] == item["policy_id"]
            assert sum(p["action"] in o for o in item["options"]) == 1
            assert p["action"] in item["options"][item["answer_idx"]]
            assert all(any(other["action"] in o for o in item["options"]) for other in policies
                       if other["category"] == p["category"]) and len(item["options"]) == 4
    assert len(corpus["style_prompts"]) == 20 and len(corpus["revoke_prompts"]) == 32
    for s in corpus["sessions"]:
        episodes = s["episodes"]
        assert 23 <= sum(e["kind"] in ("policy_intro", "policy_recur") for e in episodes) <= 25
        assert sum(len(e.get("fact_ids", [])) for e in episodes) == 60
        assert sum(len(e.get("lure_ids", [])) for e in episodes) == 20
        assert 6000 <= word_token_count("\n\n".join(e["text"] for e in episodes)) <= 8000
    assert all(count_tokens(c["text"]) <= 1536 for c in corpus["chunks"])


def census(corpus):
    counts = [len(s["episodes"]) for s in corpus["sessions"]]
    tokens = [word_token_count("\n\n".join(ep["text"] for ep in s["episodes"])) for s in corpus["sessions"]]
    return (f"policies={len(corpus['policies'])} facts={len(corpus['facts'])} lures={len(corpus['lures'])} "
            f"episodes/session={counts} tokens/session(proxy)={tokens} chunks={len(corpus['chunks'])} "
            "probes/dev=test:concept=100,verbatim=50,lure=50 style=20 revoke=32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    counter = word_token_count
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        counter = lambda text: len(tokenizer.encode(text).ids)
    corpus = build_corpus(args.seed, counter, args.sessions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(census(corpus))


if __name__ == "__main__":
    main()
