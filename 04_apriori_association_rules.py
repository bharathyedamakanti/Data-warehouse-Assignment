"""Assignment 4: Association rules using a simple Apriori implementation."""
from itertools import combinations

transactions = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "eggs"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
]

min_support = 0.4
min_confidence = 0.6
n = len(transactions)

def support(itemset):
    return sum(itemset.issubset(t) for t in transactions) / n

items = sorted(set().union(*transactions))
frequent = {}
level = [{frozenset([item]) for item in items}]
k = 1

while level[-1]:
    current = {s for s in level[-1] if support(s) >= min_support}
    if not current:
        level.pop()
        break
    frequent.update({s: support(s) for s in current})
    candidates = set()
    current_list = list(current)
    for i in range(len(current_list)):
        for j in range(i + 1, len(current_list)):
            merged = current_list[i] | current_list[j]
            if len(merged) == k + 1:
                candidates.add(merged)
    k += 1
    level.append(candidates)

print("Frequent itemsets (support):")
for itemset, sup in sorted(frequent.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    print(f"{set(itemset)}: {sup:.2f}")

print("\nAssociation rules (confidence >= %.2f):" % min_confidence)
for itemset, sup_itemset in frequent.items():
    if len(itemset) < 2:
        continue
    ordered = sorted(itemset)
    for r in range(1, len(itemset)):
        for antecedent_tuple in combinations(ordered, r):
            antecedent = frozenset(antecedent_tuple)
            consequent = itemset - antecedent
            sup_ant = frequent.get(antecedent, support(antecedent))
            confidence = sup_itemset / sup_ant if sup_ant else 0
            if confidence >= min_confidence:
                print(f"{set(antecedent)} -> {set(consequent)} "
                      f"(support={sup_itemset:.2f}, confidence={confidence:.2f})")
