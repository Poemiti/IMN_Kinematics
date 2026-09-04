# src/utils.py


def match_rule(meta, rules):
    best_match = None
    best_score = -1

    for rule in rules.get("rules"):
        conditions = rule.get("when", {})
    
        if all(meta.get(k) == v for k, v in conditions.items()):
            score = len(conditions)
            if score >= best_score:             # >= because we follow the order of the rules document, and the last one has higher priority
                best_match = rule["value"]
                best_score = score

    return best_match if best_match else rules.get("default")



