import copy,json,tempfile,unittest
from pathlib import Path
from rel01.synthetic import synthetic_answers
from rel01.data import Scope
from rel01.config import SPEC,ImplementationBlocked,RealDataForbidden
from rel01.runner import run_fit,exclusive
from rel01.storage import finish,load_final,read
from rel01.trainer import TrainingSession,tensor_digest
from rel01.authorization import Permit
ROOT=Path(__file__).resolve().parents[1]
class StorageTests(unittest.TestCase):
    def setUp(self):
        self.data,s=synthetic_answers();self.scope=Scope(s.dataset,0,None,s.fit_ids,(),s.validation_ids+s.test_ids)
    def make(self):return TrainingSession(self.data,self.scope,'M0',SPEC.seeds[0],outer_inner_epochs=[4]*4)
    def test_48_latest_two_and_final_no_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            unit=Path(d)/'fit';s=self.make()
            for _ in range(4):s.run(max_updates=s.steps_epoch);s.save(unit);self.assertLessEqual(len(list(unit.glob('checkpoint-*.pt'))),2)
            final={'binding':s.binding,'model':s.best_state,'done':True};finish(unit,final)
            self.assertFalse(list(unit.glob('checkpoint-*.pt')));self.assertNotIn('optimizer',load_final(unit));self.assertTrue((unit/'CLEANUP_COMPLETE.json').exists())
    def test_49_cleanup_failure_points_replay(self):
        for point in ['after_final_payload','after_complete','after_retire','during_cleanup']:
            with self.subTest(point=point),tempfile.TemporaryDirectory() as d:
                s=self.make();s.run();s.save(d);final={'binding':s.binding,'done':True,'model':s.best_state}
                with self.assertRaises(InterruptedError):finish(d,final,point)
                recovered=finish(d,final);self.assertEqual(tensor_digest(recovered['model']),tensor_digest(s.best_state));self.assertFalse(list(Path(d).glob('checkpoint-*.pt')))
    def test_50_foreign_and_corrupt_files_never_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'historical.txt').write_text('protected')
            with self.assertRaises(ImplementationBlocked):self.make().save(p)
            self.assertEqual((p/'historical.txt').read_text(),'protected')
        with tempfile.TemporaryDirectory() as d:
            s=self.make();s.run();s.save(d);p=Path(d);f=next(p.glob('checkpoint-*.pt'));f.write_bytes(f.read_bytes()+b'corrupt')
            with self.assertRaises(ImplementationBlocked):finish(d,{'binding':s.binding,'done':True,'model':s.best_state})
            self.assertTrue(f.exists())
    def test_51_fit_skip_resume_and_zero_epoch_relation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=(self.data,self.scope,'M0',SPEC.seeds[0])
            with self.assertRaises(InterruptedError):run_fit(*args,root/'resume',device='cpu',outer_epochs=[3]*4,interrupt_after=1)
            restored=run_fit(*args,root/'resume',device='cpu',outer_epochs=[3]*4);full=run_fit(*args,root/'full',device='cpu',outer_epochs=[3]*4)
            self.assertEqual(tensor_digest(full.best_state),tensor_digest(restored.best_state));before=(root/'resume/FINAL.pt').read_bytes()
            run_fit(*args,root/'resume',device='cpu',outer_epochs=[3]*4);self.assertEqual(before,(root/'resume/FINAL.pt').read_bytes())
            b=run_fit(self.data,self.scope,'M2',SPEC.seeds[0],root/'B0',query=full.query_artifact(),device='cpu',outer_epochs=[0]*4)
            self.assertEqual(b.updates,0);self.assertTrue((root/'B0/FINAL.pt').exists())
    def test_53_exclusive_fit_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock'
            with exclusive(p):
                with self.assertRaises(OSError):
                    with exclusive(p):pass
    def test_54_multiple_failure_records_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            unit=Path(d)/'fit'
            for epoch in [1,2]:
                with self.assertRaises(InterruptedError):run_fit(self.data,self.scope,'M0',SPEC.seeds[0],unit,device='cpu',outer_epochs=[3]*4,interrupt_after=epoch)
            self.assertEqual(len(list(unit.glob('FAILURE_*.json'))),2)
            run_fit(self.data,self.scope,'M0',SPEC.seeds[0],unit,device='cpu',outer_epochs=[3]*4)
            self.assertEqual(len(list(unit.glob('FAILURE_*.json'))),2)
