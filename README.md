# 语料审查闸门

接收版本化文档，运行文本质量规则和近重复分组，提供带租约的人工审查、纠正及受控发布。规则提示不等于人工标签。

Python 3.11 标准库与 SQLite 即可运行。数据、时钟和文件均在本地，不需要服务凭据。示例只使用合成记录。

## 运行

在仓库根目录执行：

```sh
python -m corpus_gate --db :memory: demo --fixture fixtures/demo.json
python -m examples.context
python -m examples.round_demo
python -m unittest discover -s tests -v
python -m compileall -q corpus_gate examples
```

示例使用内存或临时目录，不会改动持久台账。`examples/context.json` 记录一组可重复导入的业务输入，字段与时间口径见 `examples/数据说明.md`。`examples.context` 检查数据并调用现有产品入口；它只展示原有业务数据，不计算分析结论。`examples.round_demo` 在合成历史上演练一个完整主动审查轮次：九月人工决定进入训练与回测，十月新稿进入候选，轮次只冻结队列，实际审查仍走租约。

`tests` 覆盖现有业务生命周期、非法输入、版本和完整性边界，另有示例导入与轮次回归。完整测试和上述演示可在一次短终端会话中执行。

## 主动审查轮次

`review.round_plan` / `review.round_create`（CLI `review-round plan|create`）固定候选文档版本、去重运行、人工审查截止和分钟预算，产出一份带摘要的证据记录与队列。相关读取入口为 `review.round_get`、`review.round_evidence`、`review.round_validate`、`review-round evidence|validate|get`。

- 训练只使用截止前已最终生效的人工 accept/reject；规则 findings 仅作朴素基线，不充当真值。撤回、非当前版本和与当前状态不一致的决定列入 `training.excluded_decisions`。
- 词表、特征权重和 Platt 校准全部只在训练片段内拟合；单类标签、全未知词和全空文本分别给出 `constant_single_class`、`unknown_vocabulary`、`empty_text` 回退与常数概率。
- 回测按决定时间前推、按近重复簇隔离训练/检验，并给出 Brier、预算内三种策略的命中数/精确率/召回。只有达到配置阈值才启用文本模型；否则明确回退为去重后的规则优先，`strategy.decision` 记录原因。
- 选样按来源下限 → 固定种子探索 → 不确定性/规则优先 × 来源覆盖 ÷ 分钟排序；每簇至多一篇，预算不可超。来源下限不可行时抛出 `round_source_floor_infeasible` 并给出冲突细节，不会静默丢约束。
- 创建不改文档状态，不批准也不拒绝；结论仍由 `review.claim`/`review.decide` 租约流程产生。候选换版、已被领取、决定撤回或轮次创建后被审查都会在 `review.round_validate` 列为失效原因。同一轮重复执行返回同一队列，不重复入队。
- `review.round_evidence` 返回冻结证据摘要校验、仅凭证据复算选样的结果，以及按当前语料重放所得的摘要对照。回测命中率只描述该预算下的历史排序表现，不是总体质量估计。


## 容器

```sh
docker build -t local/corpus_gate .
docker run --rm --network none local/corpus_gate
docker run --rm --network none local/corpus_gate python -m examples.context
```

镜像构建完成后运行和测试不下载文件。持久化使用各自命令的本地路径参数；具体操作及参数可通过 `python -m corpus_gate --help` 查询。第三方依赖清单为空。
