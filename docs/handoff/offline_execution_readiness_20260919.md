# LOOK 后继远端离线接续准备（2026-09-19）

状态：design_ready_not_deployed_for_next_A。该记录落实 standing authorization 与 offline execution requirements；不是新的常驻 planner 已部署声明。

## 已经由本轮证明的能力

EmbraceNet 单种子包曾在浏览器回合中断后，由同一个远端 finite pipeline 根据已接受 receipt 自动跳过完成阶段，并连续完成剩余 residual RRR → 统计 → 报告 → verify_case。它证明“有限包的阶段链可以在一次远端启动后继续”这一能力，但不等于 Mac 完全离线时仍有通用科研 planner 常驻运行。

## 下一 A 的离线包必须自包含

ShaSpec 或后继任一批准单种子包在启动前需把以下内容全部固定到远端可读的版本化目录：

- packet/protocol/spec、科学源码 exact commit、作者来源/许可记录；
- 数据 manifest/角色与 test-sealed 声明；
- 环境/依赖锁、GPU/主机/磁盘预算；
- 训练/拟合/评价/统计 stage graph 与每 stage 的 completion receipt；
- checkpoint、私有 RNG/optimizer/scheduler/data cursor 恢复身份；
- remote manager 脚本：只领取本有限包，按 receipt 原子跳过完成阶段；
- 持久化 logs/checkpoints/result draft/final receipt；
- 停止/失败分类：可恢复资源事件与确定性科研/实现失败分开；
- 最终 self_checked_pending_independent_review 产物，不自动推 main/test。

## Mac/网页不在线时的边界

- 已经启动并通过准入的有限远端包可按其固定 stage graph 继续；
- 不需要本机文件、网页回合、SSH stdin、Mac tunnel 才能进入下一 stage；
- 不从本地复制 credential/token/private key；远端只使用已经合法配置的服务身份；
- 不因普通确认缺少用户点击而暂停，但工具/安全拒绝、数据许可变化、test 解封、不可逆外部承诺仍 fail closed；
- 没有证据表明当前存在可在 Mac 关机后自行发明新科研包的常驻 planner。科研裁决先由在线负责人登记成下一个有限 packet，再交给远端自包含执行器。

## 下一触发

next_missing_method_A_contract_20260919 的实现来源/许可/2D classification adaptation 门槛通过后，再创建一个 exact-sha remote packet；目标是启动后可独立完成“一个 A → 四棵 LOOK 树 → 统计 → report”，浏览器只负责观察/最终独立验收，不负责逐阶段续命。
