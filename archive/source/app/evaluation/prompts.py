"""No gold answers, evidence IDs, history, RAG, tools or evaluator feedback."""
PROMPT_VERSION = 'closed-book-v1'
SYSTEM_PROMPT = (
    '你正在独立完成闭卷问答。只根据题目和你自身已有知识作答。'
    '直接给出结论及必要的简短依据；不知道时明确回答不知道。'
    '涉及计算请保留单位和关键计算式，涉及关系或清单请完整列出。'
    '你没有标准答案，不得声称查看了答案或外部资料。'
)
