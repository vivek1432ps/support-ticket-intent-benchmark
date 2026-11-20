import unittest
from collections import Counter

from experiment import (LABELS, TEST, TEST_ENTITIES, TRAIN, TRAIN_ENTITIES,
                        TinyTransformer, dataset, encode, evaluate, vocabulary)


class IntentTests(unittest.TestCase):
    def test_split_is_disjoint_and_balanced(self):
        train, train_y = dataset(True)
        test, test_y = dataset(False)
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(TRAIN) & set(TEST))
        self.assertFalse(set(TRAIN_ENTITIES) & set(TEST_ENTITIES))
        self.assertEqual(set(Counter(train_y).values()), {120})
        self.assertEqual(set(Counter(test_y).values()), {24})

    def test_unknown_and_padding_are_distinct(self):
        encoded = encode(["known unseen"], vocabulary(["known"]))
        self.assertEqual(encoded.tolist()[0][:3], [2, 1, 0])
        self.assertEqual(TinyTransformer(3)(encoded).shape, (1, len(LABELS)))

    def test_end_to_end_reports_both_models(self):
        result = evaluate(1, 17)
        for name in ("tfidf_logistic_regression", "tiny_transformer"):
            metric = result[name]
            self.assertEqual(sum(map(sum, metric["confusion_matrix"])), result["test_count"])
            self.assertTrue(0 <= metric["macro_f1"] <= 1)


if __name__ == "__main__":
    unittest.main()
