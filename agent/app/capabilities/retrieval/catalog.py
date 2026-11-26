"""知识范围导航：只列出当前账号可访问的已入库资料，不生成业务事实。"""
import re

from agent.app.capabilities.retrieval.index import DEFAULT_NAMESPACE, get_index
from agent.app.capabilities.retrieval.search import chunk_allowed

LABELS = {
    "company": "公司概况与业务", "product": "产品与解决方案", "sales": "客户案例与销售",
    "support": "交付与售后服务", "tech": "技术与研发", "policy": "内部制度与流程",
    "hr": "人力资源", "finance": "经营与财务", "legal": "合同与合规",
    "org": "组织与部门职责", "strategy": "战略与经营计划", "meeting": "会议纪要",
}


def is_catalog_question(text):
    text = (text or "").strip()
    # “公司有哪些业务”属于事实问答；这里只接助手/知识库范围咨询。
    return bool(re.search(r"(知识咨询|资料范围|知识范围|咨询范围)", text)
                and re.search(r"哪些|什么|范围|覆盖|能问|能查", text)
                or "知识库" in text and re.search(r"(?:哪些|什么)(?:方面|类型|类别|主题|内容|资料|知识|文档)|覆盖.{0,6}(?:领域|主题|范围|方面|内容)", text)
                or re.search(r"(?:你|助手).{0,12}(?:能|可以).{0,10}(?:问|查|咨询).{0,8}(?:哪些|什么)(?:资料|知识|业务|内容)", text))


def visible_catalog(user):
    from agent.enterprise_rag.config import RAG_NAMESPACE
    documents = {}
    for namespace in dict.fromkeys((RAG_NAMESPACE, DEFAULT_NAMESPACE)):
        for chunk in get_index(namespace=namespace).chunks:
            if chunk_allowed(chunk, user):
                documents.setdefault((namespace, chunk["doc_id"]), chunk)
    groups = {}
    for chunk in sorted(documents.values(), key=lambda c: c.get("title", "")):
        groups.setdefault(chunk.get("category") or "other", []).append(chunk)
    examples = []
    for category in sorted(groups, key=lambda c: list(LABELS).index(c) if c in LABELS else len(LABELS)):
        for chunk in groups[category][:2]:
            examples.append({**chunk, "text": "", "section": "知识库目录", "score": 0.0})
    return examples, {"catalog": True, "documents": len(documents),
                      "categories": {c: len(items) for c, items in groups.items()}}


def render_catalog(examples, meta):
    if not examples:
        return "当前账号没有可列出的已入库资料，请先确认资料入库情况和访问权限。"
    groups = {}
    for chunk in examples:
        groups.setdefault(chunk.get("category") or "other", []).append(chunk["title"])
    lines = [f"我是 TrainPilot。当前账号可访问的已入库资料共有 {meta['documents']} 份，主要可以咨询："]
    for category, titles in groups.items():
        lines.append(f"- {LABELS.get(category, '其他已入库资料')}：" + "、".join(f"《{t}》" for t in titles) + "。")
    lines.extend(["", "你可以继续问：", f"“请介绍《{examples[0]['title']}》中的主要内容。”",
                  "也可以指定产品名称、制度主题或客户案例，我会检索资料后回答；具体答案取决于资料覆盖和访问权限。"])
    return "\n".join(lines)
