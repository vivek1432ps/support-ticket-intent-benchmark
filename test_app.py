import json
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import numpy as np

from app import DEFAULT, EXAMPLES, Handler, LIMIT, REPORT, classify
from experiment import LABELS, fit_models


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.bundle = fit_models(report["epochs"],report["seed"])
        cls.server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.server.bundle, cls.server.report = cls.bundle,report
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/"
        cls.thread = Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_predictions_are_probabilities_in_label_order(self):
        base,tiny,_,_ = classify(self.bundle,DEFAULT)
        for scores in (base,tiny):
            self.assertEqual(scores.shape,(len(LABELS),))
            self.assertTrue(np.isfinite(scores).all())
            self.assertAlmostEqual(float(scores.sum()),1,places=6)
        self.assertEqual(LABELS[base.argmax()],"refund")
        self.assertNotEqual(base.argmax(),tiny.argmax())

    def test_home_and_example_submission(self):
        with urlopen(self.url) as response:
            page = response.read().decode()
            self.assertEqual(response.headers["Cache-Control"],"no-store")
            self.assertIn("Models disagree",page)
            self.assertIn("73.3%",page)
            self.assertIn("40.0%",page)
        with urlopen(self.url,urlencode({"example":"Technical","message":"ignored"}).encode()) as response:
            page = response.read().decode()
            self.assertIn(EXAMPLES["Technical"],page)
            self.assertNotIn("ignored",page)

    def test_custom_and_invalid_input(self):
        with urlopen(self.url,urlencode({"message":"<script>refund</script>"}).encode()) as response:
            page = response.read().decode()
            self.assertIn("&lt;script&gt;refund&lt;/script&gt;",page)
            self.assertNotIn("<script>",page)
        for message, expected in [("???","short English ticket"), ("x"*(LIMIT+1),"480 characters")]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.url,urlencode({"message":message}).encode())
            self.assertEqual(caught.exception.code,400)
            self.assertIn(expected,caught.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
