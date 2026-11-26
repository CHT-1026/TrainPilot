# V7完整工具交互样例

本包从冻结V6接续LoRA SFT。以下为正式训练集的8类轨迹各1条；全部资料与人物为合成测试环境。
1280条训练=960工具轨迹+320原V6业务回放；验证160、测试160。不是同学1239题的改写。
仅assistant轮作为监督；tool返回是上下文。完整system和工具参数见contract.json。
read的offset/limit为片段序号/数量。最终引用必须来自本轮实际工具结果。

## lookup / v7-train-0264

**user**

```text
TP-TR0264的额定负载是多少？
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0264 额定负载",
        "top_k": 5
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0264额定负载为20kg。",
      "doc_id": "tr_264_spec",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
{
  "status": "answerable",
  "answer": "TP-TR0264额定负载为20kg。",
  "citations": [
    {
      "doc_id": "tr_264_spec",
      "locator": "line:1@0",
      "quote": "TP-TR0264额定负载为20kg。"
    }
  ]
}
```

## two_hop / v7-train-0865

**user**

```text
林思远的直接上级是谁，该上级负责什么岗位？
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "林思远"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "林思远的直接上级是林星川。",
      "doc_id": "tr_865_report",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
[
  {
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "林星川"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "林星川担任研发主管。",
      "doc_id": "tr_865_position",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
[
  {
    "id": "call_2",
    "type": "function",
    "function": {
      "name": "read",
      "arguments": {
        "doc_id": "tr_865_position",
        "offset": 0,
        "limit": 2
      }
    }
  }
]
```

**tool**

```json
{
  "segments": [
    {
      "locator": "line:1@0",
      "text": "林星川担任研发主管。",
      "doc_id": "tr_865_position"
    }
  ],
  "next_offset": null,
  "total_segments": 1
}
```

**assistant**

```json
{
  "status": "answerable",
  "answer": "林思远的直接上级是林星川。林星川担任研发主管。",
  "citations": [
    {
      "doc_id": "tr_865_report",
      "locator": "line:1@0",
      "quote": "林思远的直接上级是林星川。"
    },
    {
      "doc_id": "tr_865_position",
      "locator": "line:1@0",
      "quote": "林星川担任研发主管。"
    }
  ]
}
```

## empty_retry / v7-train-0258

**user**

```text
我记得旧代号是TP-TR0258-OLD，请核实这台设备的额定负载。
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0258-OLD"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": []
}
```

**assistant**

```json
[
  {
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0258 额定负载",
        "top_k": 5
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0258（旧代号TP-TR0258-OLD）的额定负载为69kg。",
      "doc_id": "tr_258_spec",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
{
  "status": "answerable",
  "answer": "TP-TR0258（旧代号TP-TR0258-OLD）的额定负载为69kg。",
  "citations": [
    {
      "doc_id": "tr_258_spec",
      "locator": "line:1@0",
      "quote": "TP-TR0258（旧代号TP-TR0258-OLD）的额定负载为69kg。"
    }
  ]
}
```

## pagination / v7-train-0155

**user**

```text
请从目录找到TP-TR0155操作规范，分别说明启动条件和停机条件。
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "list",
      "arguments": {
        "keyword": "TP-TR0155"
      }
    }
  }
]
```

**tool**

```json
{
  "documents": [
    {
      "doc_id": "tr_155_guide",
      "title": "TP-TR0155操作规范"
    }
  ],
  "next_offset": null
}
```

**assistant**

```json
[
  {
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "read",
      "arguments": {
        "doc_id": "tr_155_guide",
        "offset": 0,
        "limit": 1
      }
    }
  }
]
```

**tool**

```json
{
  "segments": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0155启动前必须完成传感器自检。",
      "doc_id": "tr_155_guide"
    }
  ],
  "next_offset": 1,
  "total_segments": 2
}
```

**assistant**

```json
[
  {
    "id": "call_2",
    "type": "function",
    "function": {
      "name": "read",
      "arguments": {
        "doc_id": "tr_155_guide",
        "offset": 1,
        "limit": 1
      }
    }
  }
]
```

**tool**

```json
{
  "segments": [
    {
      "locator": "line:2@0",
      "text": "TP-TR0155停机前必须保存作业状态。",
      "doc_id": "tr_155_guide"
    }
  ],
  "next_offset": null,
  "total_segments": 2
}
```

**assistant**

```json
{
  "status": "answerable",
  "answer": "TP-TR0155启动前必须完成传感器自检。TP-TR0155停机前必须保存作业状态。",
  "citations": [
    {
      "doc_id": "tr_155_guide",
      "locator": "line:1@0",
      "quote": "TP-TR0155启动前必须完成传感器自检。"
    },
    {
      "doc_id": "tr_155_guide",
      "locator": "line:2@0",
      "quote": "TP-TR0155停机前必须保存作业状态。"
    }
  ]
}
```

## missing / v7-train-0844

**user**

```text
资料有没有TP-TR0844未来十年的实际维修总支出？
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0844 未来十年 实际维修支出"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0844额定负载为63kg。",
      "doc_id": "tr_844_spec",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
{
  "status": "not_in_corpus",
  "answer": "当前返回的资料只包含设备参数，没有未来十年的实际维修支出，无法给出总数。",
  "citations": []
}
```

## conflict / v7-train-0573

**user**

```text
TP-TR0573同一生效日的两份记录给了不同负载，该采信哪一个？
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0573"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0573自2025-01-01生效的记录甲规定额定负载79kg。",
      "doc_id": "tr_573_a",
      "offset": 0
    },
    {
      "locator": "line:1@0",
      "text": "TP-TR0573自2025-01-01生效的记录乙规定额定负载84kg；未注明与记录甲的优先级。",
      "doc_id": "tr_573_b",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
{
  "status": "needs_clarification",
  "answer": "同日生效的记录甲为79kg，记录乙为84kg，数值冲突且无优先级，需确认适用版本，不能擅自选一个。",
  "citations": [
    {
      "doc_id": "tr_573_a",
      "locator": "line:1@0",
      "quote": "TP-TR0573自2025-01-01生效的记录甲规定额定负载79kg。"
    },
    {
      "doc_id": "tr_573_b",
      "locator": "line:1@0",
      "quote": "TP-TR0573自2025-01-01生效的记录乙规定额定负载84kg；未注明与记录甲的优先级。"
    }
  ]
}
```

## time / v7-train-0854

**user**

```text
我只关心2024年，TP-TR0854当年的标准值是多少？
```

**assistant**

```json
[
  {
    "id": "call_0",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": {
        "query": "TP-TR0854"
      }
    }
  }
]
```

**tool**

```json
{
  "hits": [
    {
      "locator": "line:1@0",
      "text": "TP-TR0854在2025年的标准值是88元。",
      "doc_id": "tr_854_history",
      "offset": 0
    }
  ]
}
```

**assistant**

```json
[
  {
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "read",
      "arguments": {
        "doc_id": "tr_854_history",
        "offset": 1,
        "limit": 1
      }
    }
  }
]
```

**tool**

```json
{
  "segments": [
    {
      "locator": "line:2@0",
      "text": "TP-TR0854在2024年的标准值是6元。",
      "doc_id": "tr_854_history"
    }
  ],
  "next_offset": null,
  "total_segments": 2
}
```

**assistant**

```json
{
  "status": "answerable",
  "answer": "TP-TR0854在2024年的标准值是6元。",
  "citations": [
    {
      "doc_id": "tr_854_history",
      "locator": "line:2@0",
      "quote": "TP-TR0854在2024年的标准值是6元。"
    }
  ]
}
```

## clarify / v7-train-0135

**user**

```text
帮我查一下那个标准。
```

**assistant**

```json
{
  "status": "needs_clarification",
  "answer": "请说明要查询的制度主题、设备型号或资料名称，我再检索对应内容。",
  "citations": []
}
```
