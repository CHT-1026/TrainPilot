"""Prepared contracts for future execution. Not run during the scaffold-only stage."""
import io
import http.client
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app.evaluation.client import ChatCompletionsClient
from app.evaluation.config import EvaluationConfig, ModelSlot
from app.evaluation.contracts import BlindQuestion, Prediction
from app.evaluation.dataset import prepare_run
from app.evaluation.files import digest, load_jsonl, write_json, write_jsonl
from app.evaluation.grading import score
from app.evaluation.runner import generate
from app.evaluation.reuse import reuse_completed


class BlindEvaluationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='blind-eval-contract-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = [{'qid': 'Q1', 'question': '负责人是谁？', 'answer_ref': 'DO_NOT_SEND',
                   'expected_doc_ids': ['SECRET_HINT'], 'question_type': 'relation', 'difficulty': 'L1', 'answer_type': 'text'},
                  {'qid': 'Q2', 'question': '总数是多少？', 'answer_ref': 'DO_NOT_SEND',
                   'expected_doc_ids': ['SECRET_HINT'], 'question_type': 'numeric', 'difficulty': 'L2', 'answer_type': 'numeric'}]
        write_jsonl(self.root/'questions.jsonl', source)
        write_jsonl(self.root/'answers.jsonl', [{'qid': row['qid'], 'question': row['question'], 'answer': answer}
                                               for row, answer in zip(source, ('张工', '合计4个'))])
        write_jsonl(self.root/'rules.jsonl', [])
        self.config = EvaluationConfig(question_bank=str(self.root/'questions.jsonl'),
            answer_bank=str(self.root/'answers.jsonl'), grading_rules=str(self.root/'rules.jsonl'),
            runs_dir=str(self.root/'runs'), seed=1, temperature=0, max_tokens=128, timeout_seconds=5,
            models={'A': ModelSlot(slot='baseline_a', base_url_env='TEST_BASE', api_key_env='TEST_KEY', model_env='TEST_MODEL')})

    def prepared(self):
        manifest = prepare_run(self.config)
        return manifest, self.root/'runs'/manifest['run_id']

    def predictions(self, directory, second_status='completed'):
        write_jsonl(directory/'predictions/A.jsonl', [
            {'qid': qid, 'status': status, 'text': text, 'error': None,
             'finish_reason': 'stop', 'request_sha256': 'fixture', 'latency_seconds': 0.1, 'usage': None}
            for qid, status, text in (('Q1', 'completed', '张工'), ('Q2', second_status, '共四个'))])
        write_json(directory/'predictions/A.seal.json', {'sha256': digest(directory/'predictions/A.jsonl')})

    def test_preparation_exports_only_qid_and_question_and_no_metrics(self):
        manifest, directory = self.prepared()
        self.assertEqual(manifest['accuracy'], {'A': None})
        for row in load_jsonl(directory/'blind/questions.jsonl'):
            self.assertEqual(set(row), {'qid', 'question'})
        self.assertEqual(list((directory/'predictions').iterdir()), [])

    def test_actual_provider_payload_has_no_answer_reference_or_tools(self):
        identity = {'base_url': 'https://provider.invalid/v1', 'model': 'fixture',
                    'temperature': 0, 'max_tokens': 128, 'timeout_seconds': 5,
                    'token_limit_parameter': 'max_tokens', 'request_options': {}}
        response = {'choices': [{'message': {'content': '不知道'}, 'finish_reason': 'stop'}]}
        with patch('app.evaluation.client.urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
            result = ChatCompletionsClient(identity, 'private-key', '共同提示词').answer(BlindQuestion(qid='Q1', question='负责人是谁？'))
        payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload['messages'], [{'role': 'system', 'content': '共同提示词'}, {'role': 'user', 'content': '负责人是谁？'}])
        self.assertNotIn('private-key', json.dumps(payload))
        self.assertNotIn('tools', payload)
        self.assertEqual(result.status, 'completed')

    def test_nonidentical_free_text_stays_pending_until_recorded_review_decision(self):
        manifest, directory = self.prepared()
        self.predictions(directory)
        first = score(self.config, manifest['run_id'], ['A'])
        self.assertIsNone(first['models']['A']['accuracy'])
        self.assertEqual(first['models']['A']['pending_review'], 1)
        item = load_jsonl(directory/'private/review_queue.jsonl')[0]
        write_jsonl(self.root/'reviews.jsonl', [{'item_id': item['item_id'], 'correct': True,
                                               'reason': '同义表达，完整涵盖所问数值', 'reviewer': 'reviewer-1'}])
        final = score(self.config, manifest['run_id'], ['A'], self.root/'reviews.jsonl')
        self.assertEqual(final['models']['A']['accuracy'], 1.0)
        self.assertEqual(load_jsonl(directory/'scores/A.jsonl')[1]['method'], 'recorded_review')

    def test_truncated_http_body_is_recorded_without_inventing_an_answer(self):
        identity = {'base_url':'https://provider.invalid/v1', 'model':'fixture', 'temperature':0,
                    'max_tokens':128, 'timeout_seconds':5, 'token_limit_parameter':'max_tokens',
                    'request_options':{}}
        response = io.BytesIO()
        with patch.object(response,'read',side_effect=http.client.IncompleteRead(b'{')), \
             patch('app.evaluation.client.urllib.request.urlopen', return_value=response):
            result = ChatCompletionsClient(identity, 'fixture-key', 'system').answer(BlindQuestion(qid='Q1',question='题目'))
        self.assertEqual((result.status, result.error, result.text), ('api_error','IncompleteRead',''))
        self.assertTrue(result.request_sha256)

    def test_json_rules_accept_key_order_but_not_extra_keys_bool_number_or_markdown(self):
        from app.evaluation.grading import grade_json
        expected={'count':1,'confirmed':False}
        self.assertTrue(grade_json('{"confirmed":false,"count":1.0}',expected)[0])
        self.assertFalse(grade_json('{"count":true,"confirmed":false}',expected)[0])
        self.assertFalse(grade_json('{"count":1,"confirmed":false,"extra":2}',expected)[0])
        self.assertFalse(grade_json('```json\n{"count":1,"confirmed":false}\n```',expected)[0])

    def test_review_snapshot_rejects_changed_raw_answer(self):
        from scripts.review_blind_answers import read_source
        source = self.root/'A_0001.jsonl'
        write_jsonl(source, [{'qid':'Q1','text':'原始回答'}])
        write_json(self.root/'A_0001.json', {'sha256':digest(source),'label':'A'})
        self.assertEqual(read_source(self.root,'A_0001')[1][0]['text'],'原始回答')
        write_jsonl(source, [{'qid':'Q1','text':'修改过的回答'}])
        with self.assertRaisesRegex(ValueError,'封存哈希'):
            read_source(self.root,'A_0001')

    def test_review_merge_uses_current_index_and_retains_old_decision_files(self):
        from scripts.review_blind_answers import merge_reviews
        queue, reviews, output = (self.root/name for name in ('queue.jsonl','reviews.jsonl','merged.jsonl'))
        write_jsonl(queue,[{'item_id':'current'},{'item_id':'still_pending'}])
        write_jsonl(reviews,[{'item_id':'old','correct':False},{'item_id':'current','correct':True}])
        before = digest(reviews)
        receipt = merge_reviews([reviews],queue,output)
        self.assertEqual(load_jsonl(output),[{'item_id':'current','correct':True}])
        self.assertEqual((receipt['included'],receipt['remaining'],receipt['model_calls']),(1,1,0))
        self.assertEqual(receipt['not_in_current_index'][0]['item_id'],'old')
        self.assertEqual(digest(reviews),before)
        with self.assertRaisesRegex(ValueError,'重复判分'):
            merge_reviews([reviews,reviews],queue,self.root/'duplicate.jsonl')
        self.assertFalse((self.root/'duplicate.jsonl').exists())

    def test_partial_scoring_keeps_resolved_decisions_in_the_merge_index(self):
        from scripts.review_blind_answers import merge_reviews
        manifest, directory = self.prepared()
        self.predictions(directory)
        score(self.config,manifest['run_id'],['A'])
        item = load_jsonl(directory/'private/review_queue.jsonl')[0]
        review_path = self.root/'review.jsonl'
        write_jsonl(review_path,[{'item_id':item['item_id'],'correct':True,
                                 'reason':'语义相同','reviewer':'fixture-reviewer'}])
        score(self.config,manifest['run_id'],['A'],review_path)
        self.assertEqual(load_jsonl(directory/'private/review_queue.jsonl'),[])
        self.assertEqual(len(load_jsonl(directory/'private/review_index.jsonl')),1)
        merged = self.root/'retained.jsonl'
        receipt = merge_reviews([review_path],directory/'private/review_index.jsonl',merged)
        self.assertEqual(receipt['included'],1)
        self.assertEqual(load_jsonl(merged),load_jsonl(review_path))

    def test_reference_change_invalidates_old_review_but_preserves_raw_predictions(self):
        manifest,directory=self.prepared()
        self.predictions(directory)
        prediction_hash=digest(directory/'predictions/A.jsonl')
        score(self.config,manifest['run_id'],['A'])
        old_item=load_jsonl(directory/'private/review_queue.jsonl')[0]
        reviews=self.root/'reviews.jsonl'
        write_jsonl(reviews,[{'item_id':old_item['item_id'],'correct':True,'reason':'旧口径复核','reviewer':'fixture'}])
        score(self.config,manifest['run_id'],['A'],reviews)
        self.attach_erratum(manifest,directory,'Q2','reference_correction','合计5个')
        with self.assertRaises(ValueError):
            score(self.config,manifest['run_id'],['A'],reviews)
        score(self.config,manifest['run_id'],['A'])
        new_item=load_jsonl(directory/'private/review_queue.jsonl')[0]
        self.assertNotEqual(old_item['item_id'],new_item['item_id'])
        self.assertEqual(digest(directory/'predictions/A.jsonl'),prediction_hash)
        self.assertGreaterEqual(len(list((directory/'scores/history').iterdir())),3)

    def valid_reuse_source(self):
        from app.evaluation.config import resolve_model
        manifest,directory=self.prepared()
        identity,_=resolve_model(self.config,'A')
        write_json(directory/'private/model_A.json',identity)
        prompt=json.loads((directory/'blind/prompt.json').read_text(encoding='utf-8'))['system']
        records=[]
        for question in load_jsonl(directory/'blind/questions.jsonl'):
            payload={'model':identity['model'],'temperature':identity['temperature'],
                     identity['token_limit_parameter']:identity['max_tokens'],'stream':False,
                     'messages':[{'role':'system','content':prompt},{'role':'user','content':question['question']}],
                     **identity['request_options']}
            records.append(Prediction(qid=question['qid'],status='completed',text='未经判分的旧原始回答',error=None,
                finish_reason='stop',request_sha256=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest(),
                latency_seconds=0.1,usage=None).model_dump())
        write_jsonl(directory/'predictions/A.jsonl',records)
        return manifest,directory

    @patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'})
    def test_reuse_is_independent_of_gold_and_skips_changed_question_without_api(self):
        source,source_dir=self.valid_reuse_source()
        source_hash=digest(source_dir/'predictions/A.jsonl')
        questions=load_jsonl(self.root/'questions.jsonl')
        answers=load_jsonl(self.root/'answers.jsonl')
        questions[0]['question']='一条修改后的新题'
        answers[0]['question']=questions[0]['question']
        answers[1]['answer']='一个新的正确参考'
        write_jsonl(self.root/'questions.jsonl',questions)
        write_jsonl(self.root/'answers.jsonl',answers)
        target,target_dir=self.prepared()
        with patch('urllib.request.urlopen') as network:
            result=reuse_completed(self.config,target['run_id'],source['run_id'],'A')
        network.assert_not_called()
        self.assertEqual((result['reused_completed'],result['not_reused']['question_text_changed']),(1,1))
        kept=load_jsonl(target_dir/'predictions/A.jsonl')[0]
        self.assertEqual((kept['qid'],kept['text']),('Q2','未经判分的旧原始回答'))
        self.assertEqual(digest(source_dir/'predictions/A.jsonl'),source_hash)

    @patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'})
    def test_reuse_rejects_changed_model_identity(self):
        source,source_dir=self.valid_reuse_source()
        target,target_dir=self.prepared()
        with patch.dict(os.environ,{'TEST_MODEL':'different-model'}):
            with self.assertRaises(ValueError):
                reuse_completed(self.config,target['run_id'],source['run_id'],'A')
        self.assertFalse((target_dir/'predictions/A.jsonl').exists())

    @patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'})
    def test_reuse_never_converts_failed_request_into_completed_answer(self):
        source,source_dir=self.valid_reuse_source()
        records=load_jsonl(source_dir/'predictions/A.jsonl')
        records[0].update(status='api_error',text='',error='HTTP_500')
        write_jsonl(source_dir/'predictions/A.jsonl',records)
        target,target_dir=self.prepared()
        result=reuse_completed(self.config,target['run_id'],source['run_id'],'A')
        self.assertEqual((result['reused_completed'],result['not_reused']['previous_interface_or_incomplete']),(1,1))

    def test_api_failure_remains_in_total_and_is_not_silently_excluded(self):
        manifest, directory = self.prepared()
        self.predictions(directory, second_status='api_error')
        result = score(self.config, manifest['run_id'], ['A'])['models']['A']
        self.assertEqual((result['total'], result['api_errors'], result['accuracy']), (2, 1, 0.5))
        self.assertEqual((result['valid_answers'], result['incorrect_answers'], result['answer_accuracy']), (1, 0, 1.0))

    def test_generation_is_disabled_without_touching_provider(self):
        with patch('app.evaluation.runner.ChatCompletionsClient') as provider:
            with self.assertRaises(ValueError):
                generate(self.config, 'not-created', 'A')
        provider.assert_not_called()

    def test_generation_report_uses_frozen_per_model_sample_sizes(self):
        from scripts import report_blind_progress as report
        manifest, directory = self.prepared()
        project = self.root/'report_project'
        target = project/'eval_bench/runs'/manifest['run_id']
        shutil.copytree(directory, target)
        sample_path = target/'blind/C_sample.json'
        write_json(sample_path, {'qids': ['Q2']})
        manifest['sampling'] = {'C': {'file': 'blind/C_sample.json',
                                     'sha256': digest(sample_path), 'total': 1}}
        write_json(target/'manifest.json', manifest)
        write_jsonl(target/'predictions/C.jsonl', [{'qid': 'Q2', 'status': 'completed'}])
        with (target/'predictions/C.jsonl').open('ab') as handle:
            handle.write(b'{"qid":"Q1"')
        output = self.root/'report_output'
        output.mkdir()
        write_json(output/'current_run.json', {'run_id': manifest['run_id'],
                   'generation_state': 'C_generating', 'labels': {'A': 'full_bank', 'C': 'sampled'}})
        before = digest(sample_path)
        with patch.object(report, 'ROOT', project), patch.object(report, 'OUT', output), \
                patch('sys.stdout', new=io.StringIO()), patch('urllib.request.urlopen') as network:
            report.main()
        network.assert_not_called()
        result = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(result['models']['A']['planned'], 2)
        self.assertEqual(result['models']['C']['planned'], 1)
        self.assertEqual(result['models']['C']['recorded'], 1)
        self.assertFalse(result['models']['C']['sealed'])
        self.assertEqual(result['grading']['eligible_planned_by_model'], {'A': 2, 'C': 1})
        self.assertEqual(digest(sample_path), before)

    def test_results_report_compares_common_questions_and_keeps_api_failure_separate(self):
        from scripts.report_blind_results import export_results
        manifest, directory = self.prepared()
        self.predictions(directory, second_status='api_error')
        for label in ['A', 'B']:
            write_json(directory/f'private/model_{label}.json', {'slot': f'fixture_{label}', 'model': 'fixture'})
        score(self.config, manifest['run_id'], ['A'])
        with self.assertRaisesRegex(ValueError, 'B尚未完成'):
            export_results(directory, ['A', 'B'], self.root/'unscored_report')
        self.assertFalse((self.root/'unscored_report').exists())
        selected = directory/'blind/B_sample.json'
        write_json(selected, {'qids': ['Q2']})
        manifest['sampling'] = {'B': {'file': 'blind/B_sample.json', 'sha256': digest(selected), 'total': 1}}
        write_json(directory/'manifest.json', manifest)
        answer = load_jsonl(directory/'predictions/A.jsonl')[1]
        answer.update(status='completed', text='合计4个')
        write_jsonl(directory/'predictions/B.jsonl', [answer])
        write_json(directory/'predictions/B.seal.json', {'sha256': digest(directory/'predictions/B.jsonl'),
                                                      'sample_sha256': digest(selected)})
        score(self.config, manifest['run_id'], ['A', 'B'])
        with patch('urllib.request.urlopen') as network:
            result = export_results(directory, ['A', 'B'], self.root/'completed_report')
        network.assert_not_called()
        self.assertEqual(result['models']['A']['accuracy'], 0.5)
        self.assertEqual(result['models']['A']['answer_accuracy'], 1.0)
        self.assertEqual(result['models']['B']['total'], 1)
        self.assertEqual(result['common_questions']['total'], 1)
        self.assertEqual(result['common_questions']['models']['A']['correct'], 0)
        self.assertEqual(result['common_questions']['models']['B']['correct'], 1)
        pair = result['pairs'][0]
        self.assertEqual((pair['compared_questions'], pair['right_wins'], pair['right_losses']), (1, 1, 0))
        self.assertEqual(pair['common_valid_answers'], {'total': 0, 'right_wins': 0, 'right_losses': 0, 'accuracy_delta': None})

    def test_results_report_requires_regrading_after_reference_erratum_changes(self):
        from scripts.report_blind_results import export_results
        manifest, directory = self.prepared()
        self.predictions(directory, second_status='api_error')
        write_json(directory/'private/model_A.json', {'slot': 'fixture_A', 'model': 'fixture'})
        score(self.config, manifest['run_id'], ['A'])
        original = digest(directory/'predictions/A.jsonl')
        self.attach_erratum(manifest, directory, 'Q2', 'reference_correction', '合计5个')
        with self.assertRaisesRegex(ValueError, '参考勘误尚未重新判分'):
            export_results(directory, ['A'], self.root/'stale_reference_report')
        self.assertFalse((self.root/'stale_reference_report').exists())
        self.assertEqual(digest(directory/'predictions/A.jsonl'), original)

    def test_pairwise_answer_comparison_does_not_treat_provider_failure_as_a_reasoning_loss(self):
        manifest, directory = self.prepared()
        self.predictions(directory, second_status='api_error')
        rows = load_jsonl(directory/'predictions/A.jsonl')
        rows[1].update(status='completed', text='合计4个')
        write_jsonl(directory/'predictions/B.jsonl', rows)
        write_json(directory/'predictions/B.seal.json', {'sha256': digest(directory/'predictions/B.jsonl')})
        result = score(self.config, manifest['run_id'], ['A', 'B'])['comparisons'][0]
        self.assertEqual(result['right_wins'], 1)
        self.assertEqual(result['common_valid_answers'],
                         {'total': 1, 'right_wins': 0, 'right_losses': 0, 'accuracy_delta': 0.0})

    def test_transport_failure_stops_before_the_remaining_questions_are_written_as_failures(self):
        manifest, directory = self.prepared()
        enabled = self.config.model_copy(update={'execution_enabled': True})
        def failed(question):
            return Prediction(qid=question.qid, status='api_error', text='', error='URLError',
                finish_reason=None, request_sha256='fixture', latency_seconds=0.1, usage=None)
        with patch.dict(os.environ, {'TEST_BASE': 'https://provider.invalid/v1', 'TEST_KEY': 'fixture', 'TEST_MODEL': 'fixture'}), \
             patch('app.evaluation.runner.ChatCompletionsClient.answer', side_effect=failed) as provider:
            with self.assertRaises(RuntimeError):
                generate(enabled, manifest['run_id'], 'A')
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(len(load_jsonl(directory/'predictions/A.jsonl')), 1)
        self.assertFalse((directory/'predictions/A.seal.json').exists())

    def test_invalid_infrastructure_campaign_cannot_be_published_as_model_accuracy(self):
        manifest, directory = self.prepared()
        manifest['status'] = 'infrastructure_invalid'
        write_json(directory/'manifest.json', manifest)
        with self.assertRaises(ValueError):
            score(self.config, manifest['run_id'], ['A'])
        self.assertFalse((directory/'summary.json').exists())

    def test_question_timeout_is_preserved_and_next_question_is_asked_once(self):
        manifest, directory = self.prepared()
        enabled = self.config.model_copy(update={'execution_enabled':True})
        calls = []

        def answer(question):
            calls.append(question.qid)
            timed_out = len(calls) == 1
            return Prediction(qid=question.qid,status='api_error' if timed_out else 'completed',
                              text='' if timed_out else '保存的实际回答',
                              error='TimeoutError' if timed_out else None,
                              finish_reason=None if timed_out else 'stop',
                              request_sha256='fixture',latency_seconds=5.0 if timed_out else 0.1,usage=None)

        with patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'}), \
             patch('app.evaluation.runner.ChatCompletionsClient.answer',side_effect=answer):
            generate(enabled,manifest['run_id'],'A')
        saved = load_jsonl(directory/'predictions/A.jsonl')
        self.assertEqual(len(set(calls)),2)
        self.assertEqual([row['status'] for row in saved],['api_error','completed'])
        self.assertEqual(saved[0]['error'],'TimeoutError')
        self.assertTrue((directory/'predictions/A.seal.json').exists())

    def test_provider_extensions_cannot_change_the_model_or_send_gold_messages(self):
        with self.assertRaises(ValueError):
            ModelSlot(slot='A', base_url_env='BASE', api_key_env='KEY', model_env='MODEL',
                      request_options={'model': 'another-model', 'messages': []})

    def attach_erratum(self, manifest, directory, qid, kind, corrected_answer):
        name = 'private/reference_errata.jsonl'
        write_jsonl(directory/name, [{'qid': qid, 'kind': kind, 'corrected_answer': corrected_answer,
                                     'reason': 'Source-backed fixture correction',
                                     'sources': [{'file': 'source-fixture.md', 'line': 1, 'quote': corrected_answer}]}])
        manifest['reference_errata'] = name
        manifest['sha256'][name] = digest(directory/name)
        write_json(directory/'manifest.json', manifest)

    def test_wrong_original_reference_is_not_accepted_after_a_source_backed_correction(self):
        manifest, directory = self.prepared()
        self.predictions(directory, second_status='api_error')
        original_hash = digest(directory/'private/answers.jsonl')
        self.attach_erratum(manifest, directory, 'Q1', 'reference_correction', '李工')
        result = score(self.config, manifest['run_id'], ['A'])['models']['A']
        row = load_jsonl(directory/'scores/A.jsonl')[0]
        self.assertEqual((row['original_reference'], row['reference'], row['verdict']), ('张工', '李工', 'pending_review'))
        self.assertEqual(result['pending_review'], 1)
        self.assertEqual(digest(directory/'private/answers.jsonl'), original_hash)

    def test_defective_question_remains_in_coverage_and_has_a_disclosed_scoring_exclusion(self):
        manifest, directory = self.prepared()
        self.predictions(directory)
        original_hash = digest(directory/'predictions/A.jsonl')
        self.attach_erratum(manifest, directory, 'Q2', 'invalid_question', '')
        result = score(self.config, manifest['run_id'], ['A'])['models']['A']
        self.assertEqual((result['total'], result['eligible_total'], result['excluded']), (2, 1, 1))
        self.assertEqual((result['correct'], result['accuracy'], result['answer_accuracy']), (1, 1.0, 1.0))
        self.assertEqual(digest(directory/'predictions/A.jsonl'), original_hash)

    def attach_sample(self, manifest, directory, label, qids):
        name = f'blind/sample_{label}.json'
        write_json(directory/name, {'qids':qids})
        manifest.setdefault('sampling', {})[label] = {'file':name,'total':len(qids),'sha256':digest(directory/name)}
        write_json(directory/'manifest.json', manifest)

    def test_fixed_sample_generation_and_score_use_its_own_denominator(self):
        manifest, directory = self.prepared()
        frozen = manifest['sha256'].copy()
        self.attach_sample(manifest,directory,'A',['Q2'])
        enabled = self.config.model_copy(update={'execution_enabled':True})
        answer = Prediction(qid='Q2',status='completed',text='合计4个',error=None,finish_reason='stop',
                            request_sha256='fixture',latency_seconds=0.1,usage=None)
        with patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'}), \
             patch('app.evaluation.runner.ChatCompletionsClient.answer',return_value=answer) as provider:
            receipt = generate(enabled,manifest['run_id'],'A')
        self.assertEqual(provider.call_args.args[0].qid,'Q2')
        self.assertEqual(provider.call_count,1)
        self.assertEqual((receipt['total'],receipt['sample_sha256']),(1,manifest['sampling']['A']['sha256']))
        result = score(self.config,manifest['run_id'],['A'])['models']['A']
        self.assertEqual((result['bank_total'],result['total'],result['accuracy']),(2,1,1.0))
        self.assertEqual({name:digest(directory/name) for name in frozen},frozen)

    def test_unequal_samples_compare_only_the_common_questions(self):
        manifest, directory = self.prepared()
        self.attach_sample(manifest,directory,'A',['Q1'])
        failed = Prediction(qid='Q1',status='api_error',text='',error='TimeoutError',finish_reason=None,
                            request_sha256='fixture',latency_seconds=5.0,usage=None).model_dump()
        correct = {**failed,'qid':'Q2','status':'completed','text':'合计4个','error':None,'finish_reason':'stop'}
        write_jsonl(directory/'predictions/A.jsonl',[failed])
        write_json(directory/'predictions/A.seal.json',{'sha256':digest(directory/'predictions/A.jsonl'),
                   'sample_sha256':manifest['sampling']['A']['sha256']})
        write_jsonl(directory/'predictions/B.jsonl',[failed,correct])
        write_json(directory/'predictions/B.seal.json',{'sha256':digest(directory/'predictions/B.jsonl')})
        result = score(self.config,manifest['run_id'],['A','B'])
        self.assertEqual((result['models']['A']['accuracy'],result['models']['B']['accuracy']),(0.0,0.5))
        pair = result['comparisons'][0]
        self.assertEqual((pair['compared_questions'],pair['accuracy_delta'],pair['right_wins']),(1,0.0,0))
        self.assertEqual(pair['common_valid_answers']['total'],0)

    def test_sample_cannot_drop_an_already_recorded_answer_on_resume(self):
        manifest, directory = self.prepared()
        self.predictions(directory)
        self.attach_sample(manifest,directory,'A',['Q1'])
        before = digest(directory/'predictions/A.jsonl')
        enabled = self.config.model_copy(update={'execution_enabled':True})
        with patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'}), \
             patch('app.evaluation.runner.ChatCompletionsClient.answer') as provider:
            with self.assertRaisesRegex(ValueError,'qid'):
                generate(enabled,manifest['run_id'],'A')
        provider.assert_not_called()
        self.assertEqual(digest(directory/'predictions/A.jsonl'),before)

    def test_changed_sample_is_rejected_before_model_request(self):
        manifest, directory = self.prepared()
        self.attach_sample(manifest,directory,'A',['Q1'])
        write_json(directory/'blind/sample_A.json',{'qids':['Q2']})
        enabled = self.config.model_copy(update={'execution_enabled':True})
        with patch.dict(os.environ,{'TEST_BASE':'https://provider.invalid/v1','TEST_KEY':'fixture','TEST_MODEL':'fixture'}), \
             patch('app.evaluation.runner.ChatCompletionsClient.answer') as provider:
            with self.assertRaisesRegex(ValueError,'抽样清单哈希'):
                generate(enabled,manifest['run_id'],'A')
        provider.assert_not_called()
