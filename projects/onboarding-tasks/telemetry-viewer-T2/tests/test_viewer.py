import unittest
from viewer import create_app

class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=create_app().test_client()

    def test_page_and_report(self):
        self.assertEqual(self.client.get("/").status_code,200)
        for route in ["/report.pdf","/track.png"]:
            with self.client.get(route) as response:
                self.assertEqual(response.status_code,200)

    def test_both_domains_and_all_laps(self):
        for query in ["lap=11&lap=12&domain=distance","lap=13&domain=time",
                      "&".join(f"lap={n}" for n in range(1,14))]:
            response=self.client.get("/plot.svg?"+query)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.mimetype,"image/svg+xml")
            self.assertIn(b"<svg",response.data)
            self.assertIn(b"Glat estimate",response.data)
            self.assertIn(b"Glong estimate",response.data)

    def test_invalid_selection(self):
        for query in ["","lap=0","lap=nan","lap=12&domain=bad"]:
            self.assertEqual(self.client.get("/plot.svg?"+query).status_code,400)
