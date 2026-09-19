- [PHD_RESEARCH_STANDARD.md](PHD_RESEARCH_STANDARD.md)：2026-09-19 用户授权后的共享科研设计/执行/留档边界；源文件 /Users/mengh/Documents/PHD/standards/research.md。
# 研究项目工作规范

LOOK 与 Radon_Bridge 使用相同源码职责、配置模式和管理命令。MHD_Framework 是独立工具箱，按安装版本选择 API，不携带本研究的服务器配置。

- 代码：`<code_root>/<project>/<timestamp>/`。
- 结果：`<data_root>/<project>/runs/<timestamp>/`。
- 环境：`<code_root>/environments/<project_lower>_<timestamp>/`。
- 共享数据由显式路径指定；共享文件不等于共享划分或 test 授权。

Ibex、ws02 是 registry.json 与 configs/deployment 中的可选实例。应用的 `scripts/manage.py preflight --profile <配置>` 支持任意服务器或本地路径；默认 local，不写死平台。

main 是通过验证的统一开发结构。新研究用时间戳标识协议与结果；修改前的旧版本在历史分支复现。GitHub 不存受限原始数据、逐参与者预测或大型检查点。

依赖从项目自己的 third_party/MHD_Framework 固定提交独立安装。当前两应用使用 V4 包，不随 MHD main 的 V5 开发自动变化。更新必须核对版本、源 SHA、输出、梯度和 state_dict；旧 Python 整对象模型仍使用原环境。

项目整理不会自动提交 GPU、运行训练或解封 test。目录检查不代表数据或实验已验收。
