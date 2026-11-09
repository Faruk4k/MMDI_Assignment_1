import matplotlib.pyplot as plt


S_min = 0.1
C_min = 0.3
list_items = ["Elixir", "Shield", "Sword", "Wand", "Giant Wand", "Gem" ]


players_transactions_dict = {
    "Player 1": ["Elixir", "Shield"],
    "Player 2": ["Gem" , "Shield"],
    "Player 3": ["Elixir", "Sword"],
    "Player 4": ["Elixir", "Wand", "Shield"],
    "Player 5": ["Elixir", "Sword"],   
    "Player 6": ["Wand", "Shield"],
    "Player 7": ["Elixir", "Wand", "Shield", "Sword"],
    "Player 8": ["Elixir", "Giant Wand"]
}

def support(itemset):
    count = 0
    for player in players_transactions_dict:
        if all(item in players_transactions_dict[player] for item in itemset):
            count += 1
    return count / len(players_transactions_dict)    

def confidence(itemset, if_items):
    # Computes confidence of the rule: if_items -> (itemset - if_items)
    # if_items   = antecedent, the "if" part of the rule (e.g. {Elixir})
    # itemset    = the full itemset, i.e. antecedent UNION consequent (e.g. {Elixir, Shield})
    # so (itemset - if_items), the leftover items, is the consequent, the "then" part
    #
    # Formula: confidence(X -> Y) = support(X union Y) / support(X)
    #   = "out of all transactions where X happened, what fraction also had Y?"
    # Here support(X union Y) is just support(itemset), since itemset already IS X union Y.

    # Sanity check: if_items must actually be part of itemset, otherwise the rule
    # doesn't make sense (can't condition on an item that isn't even in the itemset)
    for a in if_items:
        if a not in itemset:
            return 0.0

    support_itemset = support(itemset)       # support(X union Y)
    support_if_items = support(if_items)     # support(X), the antecedent alone

    # Avoid division by zero: if the antecedent never occurs, confidence is undefined,
    # so we just report 0.0 instead of crashing
    if support_itemset == 0:
        return 0.0
    if support_if_items == 0:
        return 0.0

    return support_itemset / support_if_items


def lift(itemset, if_items):
    # Computes lift of the rule: if_items -> (itemset - if_items)
    # if_items   = antecedent X, the "if" part
    # then_items = consequent Y, the "then" part, built below as itemset minus if_items
    #
    # Formula: lift(X -> Y) = confidence(X -> Y) / support(Y)
    #   = how much more often X and Y occur together than if they were independent
    # lift == 1  -> X and Y are independent (seeing X tells you nothing about Y)
    # lift  > 1  -> X and Y co-occur more than expected (positively correlated)
    # lift  < 1  -> X and Y co-occur less than expected (negatively correlated)

    for a in if_items:
        if a not in itemset:
            return 0.0
    support_itemset = support(itemset)
    support_if_items = support(if_items)
    if support_itemset == 0 or support_if_items == 0:
        return 0.0



    # Build the consequent Y by removing every item of the antecedent X from itemset
    then_items = []


    for x in itemset:
        if x not in if_items:
            then_items.append(x)

    return confidence(itemset, if_items) / support(then_items)  # confidence(X->Y) / support(Y)




def get_subsets(itemset):
    # Returns every non-empty, proper subset of itemset (i.e. excluding {} and itemset itself).
    # Each subset is a candidate antecedent for an association rule drawn from itemset.
    # Trick: every subset corresponds to a bitmask of length n = len(itemset), where
    # bit i tells us whether item i is included. mask=0 is {} and mask=2^n-1 is itemset
    # itself, so we skip those two and keep everything in between.
    n = len(itemset)
    subsets = []
    for mask in range(1, (1 << n) - 1):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(itemset[i])
        subsets.append(subset)
    return subsets


def itemsets_equal(a, b):
    return sorted(a) == sorted(b)


def contains_itemset(itemset_list, itemset):
    for existing in itemset_list:
        if itemsets_equal(existing, itemset):
            return True
    return False


def get_frequent_itemsets():
    # Level 1: every single item whose support clears S_min
    level_1 = []
    for item in list_items:
        candidate = [item]
        if support(candidate) >= S_min:
            level_1.append(candidate)

    all_frequent = list(level_1)
    current_level = level_1
    k = 2

    while current_level:
        # Candidate generation: join every pair of frequent (k-1)-itemsets whose
        # union has exactly k items (this is how Apriori builds size-k candidates
        # out of size-(k-1) frequent itemsets instead of checking every possible itemset)
        candidates = []
        for i in range(len(current_level)):
            for j in range(i + 1, len(current_level)):
                union = list(current_level[i])
                for item in current_level[j]:
                    if item not in union:
                        union.append(item)
                if len(union) == k and not contains_itemset(candidates, union):
                    candidates.append(union)

        # Pruning: a candidate can only be frequent if ALL of its (k-1)-subsets are
        # frequent (downward closure property). Discard any candidate that has an
        # infrequent subset without even having to compute its support.
        pruned_candidates = []
        for candidate in candidates:
            all_subsets_frequent = True
            for item in candidate:
                subset = [x for x in candidate if x != item]
                if not contains_itemset(current_level, subset):
                    all_subsets_frequent = False
                    break
            if all_subsets_frequent:
                pruned_candidates.append(candidate)

        # Support counting: keep only the candidates that clear S_min
        next_level = []
        for candidate in pruned_candidates:
            if support(candidate) >= S_min:
                next_level.append(candidate)

        all_frequent.extend(next_level)
        current_level = next_level
        k += 1

    return all_frequent


def get_rules(frequent_itemsets):
    # Builds every rule if_items -> then_items out of frequent itemsets of size >= 2,
    # by trying every possible non-empty, proper subset as the antecedent (if_items),
    # keeping only the rules whose confidence clears C_min.
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for if_items in get_subsets(itemset):
            then_items = [x for x in itemset if x not in if_items]
            conf = confidence(itemset, if_items)
            if conf >= C_min:
                rules.append({
                    "if_items": if_items,
                    "then_items": then_items,
                    "support": support(itemset),
                    "confidence": conf,
                    "lift": lift(itemset, if_items),
                })
    return rules


def main():
    frequent_itemsets = get_frequent_itemsets()

    print(f"Frequent itemsets (S_min = {S_min}):")
    for itemset in sorted(frequent_itemsets, key=lambda s: (len(s), -support(s))):
        print(f"  {itemset}  support = {support(itemset):.3f}")

    rules = get_rules(frequent_itemsets)

    print(f"\nAssociation rules (C_min = {C_min}):")
    for rule in sorted(rules, key=lambda r: -r["confidence"]):
        print(f"  {rule['if_items']} -> {rule['then_items']}  "
              f"support = {rule['support']:.3f}, "
              f"confidence = {rule['confidence']:.3f}, "
              f"lift = {rule['lift']:.3f}")

    # Scatter plot  x = confidence, y = support,
    # lift incorporated via marker color
    confidences = [r["confidence"] for r in rules]
    supports = [r["support"] for r in rules]
    lifts = [r["lift"] for r in rules]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(confidences, supports, c=lifts, cmap="viridis", s=100, edgecolors="black")
    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Confidence")
    plt.ylabel("Support")
    plt.title("Association rules: confidence vs. support (color = lift)")
    plt.show()


if __name__ == "__main__":
    main()
