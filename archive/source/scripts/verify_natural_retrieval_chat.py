"""Natural chat routing and within-session date correction; no /tool prefix."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_retrieval_http import OUT, request


def main():
    _, login = request('/api/auth/login', {'username':'admin','password':'admin123'})
    token = login['data']['token']
    _, session = request('/api/chat/session',{},token)
    thread = session['data']['thread_id']
    answers = []
    try:
        for question, expected in [
            ('根据澜图智能差旅费用报销标准，2025年8月1日去上海出差，一晚住宿上限多少元？请给出处。','600'),
            ('改一下，我要问的是2024年5月1日去上海的标准，不是2025年的。一晚多少元？请查当时版本并给出处。','450')]:
            status, result = request('/api/chat',{'thread_id':thread,'user_message':question},token)
            text = result.get('data',{}).get('response','')
            answers.append({'question':question,'http_status':status,'response':result,
                            'expected_value':expected,'answer_contains_expected':expected in text})
            (OUT/'natural_chat_retest.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        request('/api/chat/session/'+thread,token=token,method='DELETE')
    print(json.dumps([{'status':a['http_status'],'answer_contains_expected':a['answer_contains_expected']} for a in answers]))


if __name__=='__main__':
    main()
