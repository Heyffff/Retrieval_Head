
### 结构说明
- 问题内容：群众所问的问题
- 分数：对模型回答的评分
- 所需文档：问智检索到的文档
- 带引用的回答：模型回答，附有引用标注(如[2])
- 句子引用对齐
  - 句子：回答中的子句
  - 引用：该子句中所含的引用，支持 chunk、sentecne 或sub-sentence 粒度

### 粒度说明
#### chunk粒度
- 共有 289 条数据
- 每条问题配有 top10 检索文档，按[1]至[10]编号，每个文档整体视为一个chunk
- 引用标注仅仅标明来源文档编号（例如[2],[4]），不涉及具体位置

#### sentence 粒度
- 共有 289 条数据
- 在 chunk 粒度基础上，对每个文档内容按标点符号("。"和"；")进行划分，生成sentence 级切片
- 引用格式为[chunk_id - sentence_id]（如[1-4], [3-2]），用于定位具体语义句子

#### sub-sentence 粒度
- 共有 286 条数据（3 条因内容包含敏感信息，在 API 请求阶段被过滤）
- 在 sentence 粒度基础上，进一步使用大模型Qwen-max-latest 按语义划分为更小粒度的sub-sentence
- 每条sub-sentence 表达完整独立语义
