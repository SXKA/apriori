"""Tests for the Apriori implementation and its hash tree."""

import random
from itertools import combinations

import pytest

import main
from codec.aprior_codec import AprioriCodec
from hash_tree.hash_tree import HashTree


def brute_force_frequent_itemsets(transactions, min_support_count):
    """Enumerate all frequent itemsets by brute force.

    Args:
        transactions (list[frozenset]): Transactions.
        min_support_count (int): Minimum support count.

    Returns:
        set[frozenset]: All frequent itemsets of any size.
    """
    items = sorted(set().union(*transactions))
    result = set()
    for size in range(1, len(items) + 1):
        found = False
        for itemset in combinations(items, size):
            count = sum(1 for t in transactions if t.issuperset(itemset))
            if count >= min_support_count:
                result.add(frozenset(itemset))
                found = True
        if not found:
            break
    return result


def random_transactions(seed, n=60, n_items=10, max_len=7):
    rng = random.Random(seed)
    return tuple(
        frozenset(rng.sample(range(n_items), rng.randint(1, max_len)))
        for _ in range(n)
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("bucket_size,children_size", [(1, 2), (2, 3), (3, 5), (50, 97)])
def test_apriori_matches_brute_force(seed, bucket_size, children_size, monkeypatch):
    # Small bucket/children sizes force deep splits and heavy hash collisions.
    monkeypatch.setattr(main, "MAX_BUCKET_SIZE", bucket_size)
    monkeypatch.setattr(main, "MAX_CHILDREN_SIZE", children_size)
    transactions = random_transactions(seed)
    min_support = 0.15
    min_support_count = int(min_support * len(transactions))

    freq_itemsets, _ = main.apriori(transactions, min_support)
    got = set().union(*freq_itemsets)

    assert got == brute_force_frequent_itemsets(list(transactions), min_support_count)


def test_hash_tree_counts_support_with_deep_splits():
    candidates = [tuple(c) for c in combinations(range(6), 3)]
    tree = HashTree(itemset_size=3, max_bucket_size=1, max_children_size=2)
    tree.build(candidates)
    transaction = (0, 1, 2, 3)
    tree.update_support_count(transaction)
    # Every 3-subset of the transaction is counted exactly once.
    assert tree.find_freq_itemset(1) == {frozenset(c) for c in combinations(transaction, 3)}


def test_codec_round_trip():
    codec = AprioriCodec()
    encoded = codec.encode((frozenset({"a", "b"}), frozenset({"b", "c"})))
    decoded = codec.decode([set(encoded)])
    assert decoded == ({frozenset({"a", "b"}), frozenset({"b", "c"})},)


def test_parse_yields_last_record_without_trailing_blank_line(tmp_path):
    data = tmp_path / "Music.txt"
    data.write_text(
        "product/productId: P1\nreview/userId: U1\n\n"
        "product/productId: P2\nreview/userId: U2\n"
    )
    assert list(main.parse(str(data))) == [{"U1": "P1"}, {"U2": "P2"}]


@pytest.mark.parametrize("seed", range(5))
def test_candidate_gen_matches_pairwise_join_after_prune(seed):
    rng = random.Random(seed)
    for size in (1, 2, 3):
        freq = {frozenset(rng.sample(range(8), size)) for _ in range(25)}
        pairwise = {
            x | y for x in freq for y in freq if x != y and len(x - y) == 1
        }
        assert main.candidate_prune(main.candidate_gen(freq), freq) == main.candidate_prune(pairwise, freq)
