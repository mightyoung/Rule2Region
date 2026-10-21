"""SYNTHETIC state/JSON mutations only; no media, tensors or model execution.

Regression targets: removing strict type checks or strict JSON decoding must
accept a malformed state again and make these rejection assertions fail.
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import audit_returns as audit
import check_private_return as private

REPO = Path(__file__).resolve().parents[2]


class StrictReturn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, aggregates = audit.audit(REPO)
        cls.aggregate = aggregates['cpu-matrix']
        cls.hashes = {f: audit.digest(audit.blob(
            REPO, cls.aggregate['git_commit'], audit.BASE + f)) for f in
            ['input_manifest.json', 'model_manifest.json', 'experiment.json',
             'forward.py', 'handoff.py']}

    def fixture(self):
        return {'schema_version': 1, 'status': 'completed',
                'kind': 'real_prompt_probe', 'device': 'cpu',
                'git_commit': self.aggregate['git_commit'],
                'decision': 'unknown', 'human_GT': False,
                'manifest_hashes': copy.deepcopy(self.hashes),
                'trials': [dict(source_id=x['source_id'], arm=x['arm'],
                                decision='unknown', status='completed', exit_code=0)
                           for x in self.aggregate['trials']]}

    def test_synthetic_valid_completed_state_remains_only_state_validation(self):
        state = private.validate_return_state(self.fixture(), self.aggregate, self.hashes)
        self.assertTrue(state['successful_matrix'])
        self.assertEqual(state['recorded_trial_count'], 9)
        # This helper does not inspect bytes or grant scientific admission.

    def test_synthetic_schema_must_be_integer_not_bool_or_float(self):
        for value in [True, 1.0]:
            with self.subTest(value=repr(value)):
                state = self.fixture(); state['schema_version'] = value
                with self.assertRaisesRegex(ValueError, 'schema'):
                    private.validate_return_state(state, self.aggregate, self.hashes)

    def test_synthetic_completed_exit_must_be_integer_zero(self):
        for value in [False, True, 0.0, '0', None, 1]:
            with self.subTest(value=repr(value)):
                state = self.fixture(); state['trials'][0]['exit_code'] = value
                with self.assertRaises(ValueError):
                    private.validate_return_state(state, self.aggregate, self.hashes)

    def test_synthetic_malformed_shapes_rejected_as_value_errors(self):
        for state in [[], None, 0, {'status': []}]:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    private.validate_return_state(state, self.aggregate, self.hashes)
        for field in ['source_id', 'arm', 'status']:
            state = self.fixture(); state['trials'][0][field] = []
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    private.validate_return_state(state, self.aggregate, self.hashes)

    def test_synthetic_failed_child_integer_code_preserved(self):
        state = self.fixture(); state.update(status='failed',
            failure_type='SYNTHETIC_FAILURE', failure_reason='Synthetic fixture only')
        state['trials'] = [state['trials'][0]]
        state['trials'][0].update(status='running', exit_code=-9)
        result = private.validate_return_state(state, self.aggregate, self.hashes)
        self.assertFalse(result['successful_matrix'])
        self.assertEqual(result['trial_states'][0]['exit_code'], -9)

    def test_synthetic_bad_exit_types_rejected_in_partial_states(self):
        for status in ['running', 'failed']:
            for value in [False, 0.0, '1', None]:
                state = self.fixture(); state['status'] = status
                state['trials'] = [state['trials'][0]]
                state['trials'][0].update(status='running', exit_code=value)
                with self.subTest(status=status, value=repr(value)):
                    with self.assertRaisesRegex(ValueError, 'exit'):
                        private.validate_return_state(state, self.aggregate, self.hashes)

    def test_synthetic_running_child_without_exit_code_preserved(self):
        state = self.fixture(); state['status'] = 'running'
        state['trials'] = [state['trials'][0]]
        state['trials'][0]['status'] = 'running'; del state['trials'][0]['exit_code']
        self.assertFalse(private.validate_return_state(
            state, self.aggregate, self.hashes)['successful_matrix'])

    def test_synthetic_ambiguous_or_nonstandard_json_rejected_before_missing_files(self):
        good = json.dumps(self.fixture())
        mutations = [good.replace('"human_GT": false',
                                 '"human_GT": true, "human_GT": false'),
                     good.replace('"exit_code": 0', '"exit_code": 1, "exit_code": 0', 1)]
        mutations += [good[:-1] + ', "synthetic_unused": ' + value + '}'
                      for value in ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999']]
        for index, raw in enumerate(mutations):
            with self.subTest(synthetic_mutation=index), tempfile.TemporaryDirectory() as t:
                Path(t, 'RUN_PRIVATE.json').write_text(raw)
                with self.assertRaisesRegex(ValueError, 'duplicate JSON key|nonfinite JSON'):
                    private.inspect(REPO, Path(t), 'cpu-matrix')

    def test_synthetic_malformed_cli_writes_failure_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); run = root / 'synthetic-run'; run.mkdir()
            (run / 'RUN_PRIVATE.json').write_text('[]')
            out = root / 'synthetic-failure-report'
            proc = subprocess.run([sys.executable, str(Path(private.__file__)),
                '--repo', str(REPO), '--run', str(run), '--run-name', 'cpu-matrix',
                '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            failure = json.loads((out / 'PRIVATE_FAILURE.json').read_text())
            self.assertEqual(failure['status'], 'rejected')
            self.assertEqual(failure['decision'], 'unknown')
            self.assertEqual((run / 'RUN_PRIVATE.json').read_text(), '[]')
            self.assertFalse((out / 'PRIVATE_INTEGRITY_REPORT.json').exists())
            self.assertFalse((out / 'HUMAN_REVIEW_TEMPLATE_PRIVATE.json').exists())


if __name__ == '__main__':
    unittest.main()
