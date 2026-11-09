import matplotlib.pyplot as plt

RATINGS_PATH = "ml-1m/ratings.dat"
MOVIES_PATH = "ml-1m/movies.dat"

# D- (dislikes) is far sparser than D+ (likes): dislikes are spread thin across
# many niche movies, while likes cluster on a smaller set of popular ones, so it
# needs a much lower S_min to produce any multi-item frequent itemsets at all.
S_MIN_PLUS = 0.2
S_MIN_MINUS = 0.01
C_MIN = 0.5


# ---------------------------------------------------------------------------
# 2.1 Building the transactional databases D+ and D-
# ---------------------------------------------------------------------------

def load_ratings(path):
    # Each line: UserID::MovieID::Rating::Timestamp
    ratings = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append((int(user_id), int(movie_id), int(rating)))
    return ratings


def load_movies(path):
    # Each line: MovieID::Title::Genres -- we only need the title for display
    movies = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movies[int(movie_id)] = title
    return movies


def build_transactional_databases(ratings):
    # D_plus[user_id]  = set of movie ids that user liked    (rating 4 or 5)
    # D_minus[user_id] = set of movie ids that user disliked (rating 1 or 2)
    # Ratings of 3 are neutral and go into neither database(i think this is what is intended?).
    # Baskets are sets for quick lookup  --
    D_plus = {}
    D_minus = {}
    for user_id, movie_id, rating in ratings:
        if rating >= 4:
            if user_id not in D_plus:
                D_plus[user_id] = set()
            D_plus[user_id].add(movie_id)
        elif rating <= 2:
            if user_id not in D_minus:
                D_minus[user_id] = set()
            D_minus[user_id].add(movie_id)
    return D_plus, D_minus


def canonical_key(itemset):
    # A sorted tuple is a hashable, order-independent fingerprint of an itemset,
    # so we can put it in a set/dict and look it up in O(1) instead of doing an
    # O(n) linear scan with sorted() comparisons every time (which is what made
    # this unusably slow once there were thousands of candidate itemsets).
    return tuple(sorted(itemset))


def support(itemset, transactions, cache=None):
    # Optional memoization: the same itemset's support gets asked for again and
    # again (once per candidate during Apriori, then again for every rule drawn
    # from it), so caching by canonical_key(itemset) avoids re-scanning all of
    # `transactions` every single time. Pass a fresh {} per transactions dict.
    if cache is not None:
        key = canonical_key(itemset)
        if key in cache:
            return cache[key]

    count = 0
    for basket in transactions.values():
        if all(item in basket for item in itemset):
            count += 1
    result = count / len(transactions)

    if cache is not None:
        cache[key] = result
    return result


def confidence(itemset, if_items, transactions, cache=None):
    for a in if_items:
        if a not in itemset:
            return 0.0

    support_itemset = support(itemset, transactions, cache)
    support_if_items = support(if_items, transactions, cache)

    if support_itemset == 0:
        return 0.0
    if support_if_items == 0:
        return 0.0

    return support_itemset / support_if_items


def lift(itemset, if_items, transactions, cache=None):
    for a in if_items:
        if a not in itemset:
            return 0.0

    support_itemset = support(itemset, transactions, cache)
    support_if_items = support(if_items, transactions, cache)
    if support_itemset == 0 or support_if_items == 0:
        return 0.0

    then_items = []
    for x in itemset:
        if x not in if_items:
            then_items.append(x)

    return confidence(itemset, if_items, transactions, cache) / support(then_items, transactions, cache)


def get_subsets(itemset):
    n = len(itemset)
    subsets = []
    for mask in range(1, (1 << n) - 1):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(itemset[i])
        subsets.append(subset)
    return subsets


def get_frequent_pairs(transactions, level_1, s_min, cache=None):
    # Specialized, much faster version of the level-2 step. The generic approach
    # (generate every candidate pair from level_1, then scan all transactions for
    # each one) costs O(pairs * transactions) -- with hundreds of frequent single
    # items and thousands of transactions, that is billions of operations.
    # Instead, make a single pass over the transactions and count every
    # co-occurring pair of frequent items directly. This costs roughly
    # O(sum of basket_size^2), which is far smaller since most items in a basket
    # are not even in level_1, and is exactly how one would count pairs by hand.
    level_1_items = set(itemset[0] for itemset in level_1)

    pair_counts = {}
    for basket in transactions.values():
        basket_frequent_items = sorted(item for item in basket if item in level_1_items)
        for i in range(len(basket_frequent_items)):
            for j in range(i + 1, len(basket_frequent_items)):
                pair = (basket_frequent_items[i], basket_frequent_items[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    n = len(transactions)
    frequent_pairs = []
    for (a, b), count in pair_counts.items():
        pair_support = count / n
        if pair_support >= s_min:
            itemset = [a, b]
            frequent_pairs.append(itemset)
            if cache is not None:
                cache[canonical_key(itemset)] = pair_support

    return frequent_pairs


def get_frequent_itemsets(transactions, items, s_min, cache=None):
    level_1 = []
    for item in items:
        candidate = [item]
        if support(candidate, transactions, cache) >= s_min:
            level_1.append(candidate)

    all_frequent = list(level_1)

    if not level_1:
        return all_frequent

    current_level = get_frequent_pairs(transactions, level_1, s_min, cache)
    all_frequent.extend(current_level)
    current_level_keys = set(canonical_key(itemset) for itemset in current_level)
    k = 3

    while current_level:
        # Candidate generation, deduplicated via canonical_key() instead of a linear scan
        candidates = []
        seen_candidate_keys = set()
        for i in range(len(current_level)):
            for j in range(i + 1, len(current_level)):
                union = list(current_level[i])
                for item in current_level[j]:
                    if item not in union:
                        union.append(item)
                if len(union) == k:
                    key = canonical_key(union)
                    if key not in seen_candidate_keys:
                        seen_candidate_keys.add(key)
                        candidates.append(union)

        # Pruning: a candidate can only be frequent if ALL of its (k-1)-subsets are
        # frequent (downward closure property) -- checked via O(1) set lookups.
        pruned_candidates = []
        for candidate in candidates:
            all_subsets_frequent = True
            for item in candidate:
                subset_key = canonical_key([x for x in candidate if x != item])
                if subset_key not in current_level_keys:
                    all_subsets_frequent = False
                    break
            if all_subsets_frequent:
                pruned_candidates.append(candidate)

        # Support counting: keep only the candidates that clear S_min
        next_level = []
        for candidate in pruned_candidates:
            if support(candidate, transactions, cache) >= s_min:
                next_level.append(candidate)

        all_frequent.extend(next_level)
        current_level = next_level
        current_level_keys = set(canonical_key(itemset) for itemset in current_level)
        k += 1

    return all_frequent


def get_rules(frequent_itemsets, transactions, c_min, cache=None):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for if_items in get_subsets(itemset):
            then_items = [x for x in itemset if x not in if_items]
            conf = confidence(itemset, if_items, transactions, cache)
            if conf >= c_min:
                rules.append({
                    "if_items": if_items,
                    "then_items": then_items,
                    "support": support(itemset, transactions, cache),
                    "confidence": conf,
                    "lift": lift(itemset, if_items, transactions, cache),
                })
    return rules


def titled(itemset, movies):
    # Renders a list of movie ids as "{Title A, Title B}" for display
    return "{" + ", ".join(movies.get(m, f"Unknown movie {m}") for m in itemset) + "}"


def print_itemsets_and_rules(label, frequent_itemsets, rules, movies):
    print(f"\n=== {label} ===")

    print(f"Frequent itemsets ({len(frequent_itemsets)}):")
    for itemset in sorted(frequent_itemsets, key=len):
        print(f"  {titled(itemset, movies)}")

    print(f"\nAssociation rules ({len(rules)}):")
    for rule in sorted(rules, key=lambda r: -r["confidence"]):
        print(f"  {titled(rule['if_items'], movies)} -> {titled(rule['then_items'], movies)}  "
              f"support = {rule['support']:.3f}, "
              f"confidence = {rule['confidence']:.3f}, "
              f"lift = {rule['lift']:.3f}")


def plot_rules(label, rules):
    confidences = [r["confidence"] for r in rules]
    supports = [r["support"] for r in rules]
    lifts = [r["lift"] for r in rules]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(confidences, supports, c=lifts, cmap="viridis", s=60, edgecolors="black")
    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Confidence")
    plt.ylabel("Support")
    plt.title(f"{label}: confidence vs. support (color = lift)")
    plt.show()


def main():
    ratings = load_ratings(RATINGS_PATH)
    movies = load_movies(MOVIES_PATH)
    D_plus, D_minus = build_transactional_databases(ratings)

    items_plus = list(set().union(*D_plus.values()))
    items_minus = list(set().union(*D_minus.values()))

    cache_plus = {}
    freq_plus = get_frequent_itemsets(D_plus, items_plus, S_MIN_PLUS, cache_plus)
    rules_plus = get_rules(freq_plus, D_plus, C_MIN, cache_plus)
    print_itemsets_and_rules(f"D+ (liked movies, S_min={S_MIN_PLUS}, C_min={C_MIN})", freq_plus, rules_plus, movies)

    cache_minus = {}
    freq_minus = get_frequent_itemsets(D_minus, items_minus, S_MIN_MINUS, cache_minus)
    rules_minus = get_rules(freq_minus, D_minus, C_MIN, cache_minus)
    print_itemsets_and_rules(f"D- (disliked movies, S_min={S_MIN_MINUS}, C_min={C_MIN})", freq_minus, rules_minus, movies)

    plot_rules("D+ rules", rules_plus)
    plot_rules("D- rules", rules_minus)


if __name__ == "__main__":
    main()
