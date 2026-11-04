# TR-V2 / K=1 — DESIGN_REL01_V1.1 实施合同（审核修订版）

> **日期**：2026-10-21  
> **本版本状态**：`IMPLEMENTATION_SPEC_READY`（允许进入代码实现与合成验收；**不授权真实监督训练、真实模型预测或解封评价**）。  
> **来源**：`TRV2_K1_DESIGN_REL01_V1_AUDIT_DELIVERY.zip` 中 `DESIGN_REL01_AUDIT_REPORT_CN.md`、`MINIMUM_REQUIRED_AMENDMENTS.md` 等；本文件是审核后作出的科学裁决，新增的具体默认值属于**待工程验收的实施合同**，不是审计已实测的性能结果。  
> **版本关系**：保留V1网络科学结构，补齐8项P1实施缺口；不恢复已搁置的旧R01-Anchored DESIGN06。旧PRL/TPF不混入。

## 1. 研究目标和权限

**目标**：仅一条本人已知诚实回答 `r`，检验可学习的语义参照关系能否在Query-only基础上提高欺骗判别，辨别本人条件收益、非本人语义信息收益及DOLOS来源/MU3D任务条件的混杂。任何结果均属反复探索过的数据上的探索性验证，不宣称已经识别心理欺骗因果机制。

- DOLOS：41人/599回答；MU3D：80人/320回答；严格分数据集拟合，沿用现有参与者独立外五折/内四折与K=1角色表。
- 冻结GTE CLS768及Core36原值/mask，**不重做ASR、编码或特征提取**。
- 允许Codex先实施工程代码、合成单元测试和只读资产绑定检查；未明确再授权前，禁止任何真实监督拟合、自动正式训练或外测预测。

## 2. 唯一模型前向图

`e_q,e_r ∈ R^768`，均取原冻结GTE CLS、L2标准化。所有Linear均`bias=True`；GELU用框架精确模式（非tanh近似）；LayerNorm `eps=1e-5, elementwise_affine=True`；各分支 **Linear → GELU → LayerNorm**，无Dropout、无BatchNorm。

Query语义：`h_q = LN64(GELU(Linear(768,64)(e_q)))`。

Query Psy：`x_P=[b_scaled_36, m_effective_36]`；`p_q=LN16(GELU(Linear(72,16)(x_P)))`。

Query Head：
- M0: `Linear(64,32) → GELU → Linear(32,1)`；
- M1: `Linear(80,32) → GELU → Linear(32,1)`，输入 `[h_q,p_q]`。

Reference：独立于Query投影的`P_R = Linear(768,64) → GELU → LN64`；同一组 `P_R` 权重用于`q/r/c`：

```
u_q=P_R(e_q); u_r=P_R(e_r); u_c=P_R(c)
phi(q,r)=[u_q, u_r, abs(u_q-u_r), u_q*u_r, dot(e_q,e_r)]  # 257维
phi(q,c)=[u_q, u_c, abs(u_q-u_c), u_q*u_c, dot(e_q,c)]
F(q,x)=Linear(257,32) -> GELU -> Linear(32,1) applied to phi(q,x)
R(q,r)=F(q,r)-F(q,c)
z(q,r)=z_Q(q)+R(q,r)
```

所有原CLS已经L2归一化，`c`也必须L2归一化，因此dot是原CLS的余弦；如读入范数不合格或中心归一化分母过小，报告BLOCK而非静默修复。输出是logit；Sigmoid仅用于概率报告。

`c`按当前fit作用域的每人诚实回答均值，再对训练参与者等权平均、L2归一化；当前scope的训练参与者可以出现在自己的训练中心内。验证/测试身份不得参与中心；训练侧用同一scope固定中心，不偷偷变成leave-p-out。保存原CLS中心及其源ID/SHA；**不detach**关系分支中的`F(q,c)`或`P_R(c)`，不缓存随参数变化的F/P_R输出。

Relation末层含bias以保留精确审计参数数，但在`F(q,r)-F(q,c)`中恒抵消，其数据梯度应为零。Relation末层weight初始为0，bias初始为0；Relation其它层正常种子初始化。首步关系上游梯度为0是可预期现象，必须在一次有效更新后验证上游梯度可非零。

参数表（PyTorch实例化须一致）：M0=51,457；M1=53,169；M2/C1-0=109,090；M3/C1-1=110,802。Relation共57,633；GTE冻结权重不计入可训练量。

## 3. 样本与训练目标：两阶段分布统一

对于训练参与者`p`，其N条唯一回答中T条诚实、L=N−T条欺骗。其每条唯一查询回答的K=1**边际权重**：

```
w_in_person(q) = (T-1)/(T*(N-1))  if q is truthful
               = 1/(N-1)          if q is deceptive
w_Q(q) = w_in_person(q)/number_of_train_participants
```

核对每名参与者`sum_q w_in_person(q)=1`及全scope`sum_q w_Q(q)=1`。MU3D的权重应使欺骗总质量为2/3（非原唯一回答的1/2）。

**阶段A**：仅训练M0/M1 Query参数；仍遍历训练身份唯一回答，不把展开角色当作独立样本；目标`sum_q w_Q(q)*BCEWithLogits(z_Q(q),y_q)`。

**阶段B**：冻结对应同一scope、seed的Query参数、令Query `eval()` 并排除优化器，仅训练Relation；损失按训练参与者→本人诚实参照→原合法查询逐级等权的`BCEWithLogits(z_Q(q)+R(q,r),y_q)`；M2与C1-0必须复用同一M0，M3与C1-1复用同一M1。对C1仅把`r_self`替换成合法`o`，**不改变原`(p,r_self,q)`角色集合**。

六模型：M0、M1、M2(SELF/无Psy)、M3(SELF/查询Psy)、C1-0(OTHER/无Psy)、C1-1(OTHER/查询Psy)。M3没有参照端Psy或显式Psy×语义关系，不能宣称学到了个人Psy变化。

## 4. 完整Psy数值合同

- 使用现有Core36原始类型、mask和reason；不重造分母。每列仅在当前fit训练身份、**有效值**中，用上面的`w_Q`重新归一求`mu`和`var`，使用float64累积；有效值包含合法的0。
- `sigma=sqrt(var)`；若无训练有效值：`fit_available=0`, 该列有效输入mask强制为0，标准化数值为0；若`var<1e-12`：固定`scale=1`并记`NEAR_CONSTANT`，否则`scale=sigma`。该阈值是预先固定的**工程数值选择**，非已获验证的最优值。
- 若有训练有效值，则`m_effective = raw_mask AND fit_available`。有效输入`b_scaled=(raw-mu)/scale`，无效占位0，不能执行`NaN*0`。模型输出`float32`；原值与raw mask/reason完整保留。输入严格`36+36=72`维，不隐式增加词长支持量或其它通道。
- 禁止用验证/外测身份计算mu/var、改变全缺失列策略、用外测成绩调剪裁/标准化。若合法值经处理仍NaN/Inf，直接阻断。

## 5. 冻结采样、优化与Epoch

### 阶段A
- 每Epoch对当前fit训练身份全部唯一回答使用确定性seed控制的无放回Shuffle；batch=32，最后一批使用真实样本数`b`。
- 每batch损失：`(N_unique_fit/b)*sum_{q in batch}[w_Q(q)*BCE(q)]`。批内**不得再按sum(w_Q)归一**，避免目标错位。
- 每Epoch步数 `ceil(N_unique_fit/32)`，最多100 Epoch。

### 阶段B
- 每step均匀选8名**不同**训练参与者（步间可重复）；每人均匀选1条合法诚实参照，再从该参照原合法查询集合中**有放回**均匀取4条。每步32角色；均值loss，无额外角色权重（已通过抽样保证人/参照/查询三级等权）。
- 每Epoch步数 **`S_fit=ceil(number_of_legal_truth_reference_conditions_in_fit / 8)`**；最多80 Epoch。MU3D每r仅3条查询，抽4次的重复必须被允许并记录，不伪装成4个独立样本。
- SELF和对应C1分支共享同seed的初始Relation参数、`(p,r_self,q)`抽样流，仅将关系输入r替换为冻结OTHER供体；不根据标签筛选OTHER。

### 通用训练值（单一候选，不搜索）
- Query：AdamW lr=3e-4，WD=1e-3，patience=15。
- Relation：AdamW lr=1e-3，WD=1e-3，patience=12。
- AdamW `betas=(0.9,0.999), eps=1e-8`；矩阵weight启用WD，bias与LayerNorm仿射参数不做WD；无scheduler。
- 梯度裁剪global norm=1.0；float32，不启用AMP/TF32；确定性seed控制Python/NumPy/Torch/采样器，不加入Dropout。
- 固定训练seed `[20260722,20260723,20260724]`；执行环境/算法非确定性如有必须记录并停在工程验收，不能伪称bit-exact。

## 6. OTHER构造与严格作用域

Role主键固定`(dataset,outer_fold,inner_fold_or_refit,p,r_self,q)`，`q!=r_self`。OTHER候选为相应**fit训练参与者**中`other_p!=p`且已知诚实的唯一回答。内层训练及验证的OTHER候选均只来自inner TRAIN；外层最终训练及测试候选均只来自outer TRAIN。测试参与者的其他诚实回答或欺骗标签绝不用于选择OTHER。

按`abs(log1p(Nlex(o))-log1p(Nlex(r_self)))`最小匹配；平局按稳定的`SHA256(20260722|scope|p|r_self|o)`排序，最后以ID排序作为最终确定性平局处理。同一scope中所有seed、M/C1及固定权重反事实共用实际`other_id`。若无候选，标记`UNMATCHED`并在需要OTHER的**共同合法子集**上进行比较，不将SELF全量减去OTHER子集均值。DOLOS严格同来源OTHER为固定模型换参照的敏感性分析，不计入六模型额外重训。

## 7. 内层模型选择、外层重拟合与提前停止

- 内层每一个fit/validation身份作用域分别拟合Query、Normalizer、诚实中心、Relation、OTHER清单；验证仅调用该fit派生的对象。
- Query与Relation均在对应K=1合法角色上计算**参与者宏BCE**：先一条参照条件内平均合法q的BCE，再对该人参照等权，再跨验证参与者等权。Query虽无参照，也按相同角色权重验证。
- 每Epoch结束验证；Query epoch从1至100选择，Relation先记录**epoch0零残差**再观察epoch1至80，允许最佳Relation epoch=0（表示关系不值得启用）。
- 更佳判据：当前val宏BCE严格小于best减`1e-6`；差值不足或相等一律保持最早best，累计patience；best model来自符合条件的最早epoch。
- 每dataset×分支×seed×outer独立，从四inner最佳Epoch取中位数并**向上取整**：Query给出A外层拟合Epoch，Relation给出B外层拟合Epoch；outer TRAIN从头重拟合，B冻结该outer新训Query，按定量Epoch固定训练，不用outer TEST早停。若B=0则明确记录无关系更新，仍保留计划槽位。不得将inner模型/Normalizer带入outer。
- 早停指标宏BCE，评价AP/AUROC不用于调epoch/阈值/方向。

## 8. 评价、消融和科学判断

- 冻结K=1角色集合，DOLOS与MU3D独立。每个`(参与者p,参照r,seed)`计算AP/AUROC/BCE及分类指标；先在人内等权参照、再人内等权seed，最后参与者宏。**不**将三个seed的logit/prob先平均成ensemble；DOLOS不以五折等权平均替代41人等权。
- AP是**主排序指标**，AUROC为重要支持排序指标；并完整报告ACC、Balanced ACC、欺骗F1、Precision、Recall、BCE，分类阈值固定`logit>=0`。并列分数按精确同分组规则计算AP/AUROC，不加随机jitter；浮点近似并列额外审计，但不对真实预测隐式重排。
- 主要配对比较：M2−M0、M3−M1（参照增量）；M1−M0（查询Psy）；M2−C1-0、M3−C1-1（已训练非本人控制），DOLOS/MU3D分别报告。每组严格同p/r/q/seed集合；参与者级5,000次配对Bootstrap，保留同人所有r/seed相关性。区间默认未做多重比较校正；属于探索性证据，**不**称确认性显著性。
- 已训Relation模型分别按SELF、`c`-NULL、实际OTHER及DOLOS同源OTHER可行子集进行固定权重反事实。必须报告SELF、NULL、OTHER各自绝对AP/AUROC/BCE、相同r下真假margin、严格翻转、Rescue/Damage、共同偏移与中心化差异，并记录train/val基础BCE及残差分布。
- 预先区分三个结论等级：`QUERY_GAIN_ONLY`（仅查询信息增量）、`CONDITIONAL_GAIN`（SELF相对Query增量但本人专属性不足）、`PERSON_SPECIFIC_CONDITIONAL_EVIDENCE`（还需优于已训OTHER控制，固定权重排序净收益合理，且如实披露混杂）。后者也**不能**推出个人心理机制因果识别。
- MU3D按第四批PT/NT/PL/NL条件展开，不能独立隔离效价/对象/固定顺序；DOLOS按来源及可匹配同源OTHER敏感性分析。历史R01只作不同训练方式/seed限制下的强基线，不混用不同实际OTHER控制。

## 9. 计划拟合预算（**尚未获运行授权**）

- 每数据集：inner `5×4×3×(2 Query +4 Relation)=360`，outer `5×3×6=90`，总450。
- 两数据集：**720 inner + 180 outer = 900计划拟合槽位**，Query复用已计算在内。
- 最新已知752容器无可见GPU、0.5核CPU、2GiB可用内存；实时状态/性能不得假设不变。真实吞吐、显存、耗时均UNKNOWN，需要将来的合成工作负载实测。

## 10. Codex下一步实现范围与验收门槛

**允许实现**：PyTorch模型结构、数据只读装载与SHA验证、Psy安全Normalizer、K1角色索引和OTHER冻结选择、参与者等权采样、训练器接口（但禁止真实数据训练）、独立评价器、合成及数学单元测试、日志/manifest与resume协议、CPU/GPU合成资源基准。

**必须通过的零真实监督训练验收**：

1. M0/M1/M2/M3/C1精确参数量、形状、finite、逐分支开关；
2. `z(q,c)=z_Q(q)`在fp32推理容差 `atol=1e-6, rtol=1e-6`内成立；非退化合成参照确有改变同r查询排序的**表达能力**（构造权重测试，不声称训练效果）；
3. `F(q,c)`及其关系投影参与梯度；Relation零末层首步和下一步梯度行为正确；Query在B阶段`eval`且不更新；
4. MU3D角色边际权重欺骗质量为2/3，所有合法角色权重与逐ID枚举一致；采样梯度在固定参数下与完整目标Monte Carlo近似一致；
5. OTHER替换不改变任何`q`/角色ID；禁止跨身份/跨inner作用域供体；中心与Normalizer绝无验证/测试身份；
6. 缺失/真实0/全缺失列/近零方差/NaN安全处理均有断言；
7. 内层best epoch、epoch0、同分、outer refit、断点恢复、seed隔离与评价汇总通过合成测试；
8. 演示共同位移不会提高固定参照AP/AUROC、而关系确实具有非加性影响排序的能力；
9. 工程测试不修改任何冻结历史资产，运行前后关键SHA一致；
10. 输出资源峰值、设备信息、合成吞吐以及尚未核实的环境限制；**禁止启动任何真实监督拟合/真实测试预测**。

**交付状态**：`IMPLEMENTATION_SMOKE_PASS` / `IMPLEMENTATION_BLOCKED`；报告发现问题和对应最小修改。必须得到用户后续明确批准才可冻结正式运行并执行真实训练。

