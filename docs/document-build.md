# 文档生成与检查

`product-plan.md` 是权威源，Word 副本由 `scripts/build_project_doc.py` 生成。脚本依赖 python-docx 和 Pillow，架构图使用 Windows 字体路径；在其他操作系统运行时需将脚本内的字体路径指向可用中文字体。

生成方式：

```bash
python scripts/build_project_doc.py
```

Markdown 内包含七个显式分页标记，Word 按八个章节页设计。显式分页不能保证最终恰好八页，实际页数必须通过文档渲染核对。

当前文件已通过 OOXML 包结构、内容和链接检查，并检查了架构图。配套 render_docx.py 因缺少 LibreOffice soffice.exe 无法生成页面 PNG；Word 兼容转换接口也不可用。因此尚未完成 Word 视觉校验，不能把设计页数当成已验证页数。

具备文档渲染器后，渲染全部页面并检查字体、表格、分页、留白和链接。调整后重新生成并再次检查，最终将 Markdown 与 Word 在同一提交中更新。本地渲染文件保存在忽略的 .qa/ 目录。
