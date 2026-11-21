import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.common import ROOT,normalize

class TextHandlingTests(unittest.TestCase):
    def test_preserves_evidence(self):
        self.assertEqual(normalize('  Қ*ТАҚ!\nКүшті  '),'қ*тақ! күшті')
    def test_masks_contact_details(self):
        s=normalize('test@example.com +7 777 123 4567 @reviewer https://example.com/x')
        for expected in ['[email]','[number]','[user]','[url]']:self.assertIn(expected,s)
        self.assertNotIn('123 4567',s)
    def test_unicode_and_empty(self):
        self.assertEqual(normalize('ＡＢＣ'),'abc')
        self.assertEqual(normalize('  '),'')

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(ROOT/'data/labeled.csv',keep_default_na=False)
    def test_groups_do_not_cross_splits(self):
        self.assertEqual(self.df.groupby('group_id').split.nunique().max(),1)
    def test_no_exact_duplicates(self):
        self.assertTrue(self.df.text.map(normalize).is_unique)
    def test_contrast_pairs_stay_together(self):
        d=self.df[self.df.is_synthetic.eq(1)]
        self.assertTrue(d.groupby('source_id').split.nunique().eq(1).all())
    def test_both_labels_in_each_split(self):
        for _,d in self.df.groupby('split'):self.assertEqual(set(d.label),{0,1})
    def test_annotated_filter_miss_is_positive(self):
        d=self.df[self.df.id.eq('real_pla120419pla')]
        self.assertEqual(int(d.iloc[0].label),1)
    def test_documented_near_duplicate_limit(self):
        summary=json.loads((ROOT/'data/dataset_summary.json').read_text())
        self.assertLess(summary['max_cross_split_similarity'],summary['near_duplicate_threshold'])
    def test_normalized_text_is_reproducible(self):
        self.assertEqual(self.df.normalized_text.tolist(),self.df.text.map(normalize).tolist())
    def test_requested_composition(self):
        self.assertEqual(len(self.df),2000)
        self.assertEqual(int(self.df.is_synthetic.eq(0).sum()),1400)
        self.assertEqual(int(self.df.is_synthetic.eq(1).sum()),600)
    def test_previous_test_never_enters_development(self):
        prior=pd.read_csv(ROOT/'data/previous_split_membership.csv')
        old=set(prior[prior.split.eq('test')].id)
        self.assertTrue(old.issubset(set(self.df[self.df.split.eq('test')].id)))
    def test_new_test_has_no_previously_seen_ids(self):
        prior=pd.read_csv(ROOT/'data/previous_split_membership.csv')
        new=set(self.df[self.df.evaluation_cohort.eq('new_test')].id)
        self.assertTrue(new.isdisjoint(set(prior.id)))
    def test_template_families_stay_together(self):
        d=self.df[self.df.selection.eq('template_substitution')].copy()
        d['template']=d.source_id.str.extract(r'(v2_template_\d+)')[0]
        self.assertTrue(d.groupby('template').split.nunique().eq(1).all())

class InferenceTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'artifacts/model.joblib').exists(),'Train model first')
    def test_saved_model_score_matches_sigmoid(self):
        from src.predict import predict
        p=predict('Тауар сапасы жақсы.')
        sigmoid=1/(1+np.exp(-p['decision_score']))
        self.assertAlmostEqual(p['profanity_probability'],sigmoid,places=10)
        self.assertEqual(p['prediction'],int(p['profanity_probability']>=p['threshold']))
    @unittest.skipUnless((ROOT/'artifacts/model.joblib').exists(),'Train model first')
    def test_empty_input_rejected(self):
        from src.predict import predict
        with self.assertRaises(ValueError):predict('  ')

if __name__=='__main__':unittest.main()
