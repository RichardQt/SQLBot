<p align="center"><img src="https://resource-fit2cloud-com.oss-cn-hangzhou.aliyuncs.com/sqlbot/sqlbot.png" alt="SQLBot" width="300" /></p>
<h3 align="center">基于大模型和 RAG 的智能问数系统</h3>
<p align="center">
  <a href="https://github.com/dataease/SQLBot/releases/latest"><img src="https://img.shields.io/github/v/release/dataease/SQLBot" alt="Latest release"></a>
  <a href="https://github.com/dataease/SQLBot"><img src="https://img.shields.io/github/stars/dataease/SQLBot?color=%231890FF&style=flat-square" alt="Stars"></a>    
  <a href="https://hub.docker.com/r/dataease/SQLbot"><img src="https://img.shields.io/docker/pulls/dataease/sqlbot?label=downloads" alt="Download"></a><br/>

</p>
<hr/>

SQLBot 是一款基于大模型和 RAG 的智能问数系统。SQLBot 的优势包括：

- **开箱即用**: 只需配置大模型和数据源即可开启问数之旅，通过大模型和 RAG 的结合来实现高质量的 text2sql；
- **易于集成**: 支持快速嵌入到第三方业务系统，也支持被 n8n、MaxKB、Dify、Coze 等 AI 应用开发平台集成调用，让各类应用快速拥有智能问数能力；
- **安全可控**: 提供基于工作空间的资源隔离机制，能够实现细粒度的数据权限控制。

## 工作原理

<img width="1105" height="577" alt="system-arch" src="https://github.com/user-attachments/assets/462603fc-980b-4b8b-a6d4-a821c070a048" />

## 快速开始

### ⚠️ 重要安全提示

在生产环境部署前，**请务必完成安全配置**，否则系统将面临严重的安全风险！

🔐 **必须立即修改的配置项**：

1. **JWT 密钥** - 生成唯一的 SECRET_KEY（详见 [SECURITY.md](./SECURITY.md)）
2. **数据库密码** - 修改默认的 PostgreSQL 密码
3. **管理员密码** - 首次登录后立即修改默认密码

📖 **完整安全配置指南**：请阅读 [SECURITY.md](./SECURITY.md) 文档

### 安装部署

准备一台 Linux 服务器，安装好 [Docker](https://docs.docker.com/get-docker/)，执行以下一键安装脚本：

```bash
docker run -d \
  --name sqlbot \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8001:8001 \
  -v ./data/sqlbot/excel:/opt/sqlbot/data/excel \
  -v ./data/sqlbot/file:/opt/sqlbot/data/file \
  -v ./data/sqlbot/images:/opt/sqlbot/images \
  -v ./data/sqlbot/logs:/opt/sqlbot/app/logs \
  -v ./data/postgresql:/var/lib/postgresql/data \
  --privileged=true \
  dataease/sqlbot
```

你也可以通过 [1Panel 应用商店](https://apps.fit2cloud.com/1panel) 快速部署 SQLBot。

如果是内网环境，你可以通过 [离线安装包方式](https://community.fit2cloud.com/#/products/sqlbot/downloads) 部署 SQLBot。

### 访问方式

- 在浏览器中打开: http://<你的服务器 IP>:8000/
- 用户名: admin
- 密码: SQLBot@123456

### 联系我们

如你有更多问题，可以加入我们的技术交流群与我们交流。

<img width="180" height="180" alt="contact_me_qr" src="https://github.com/user-attachments/assets/2594ff29-5426-4457-b051-279855610030" />

## UI 展示

  <tr>
    <img alt="q&a" src="https://github.com/user-attachments/assets/55526514-52f3-4cfe-98ec-08a986259280"   />
  </tr>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dataease/sqlbot&type=Date)](https://www.star-history.com/#dataease/sqlbot&Date)

## 飞致云旗下的其他明星项目

- [DataEase](https://github.com/dataease/dataease/) - 人人可用的开源 BI 工具
- [1Panel](https://github.com/1panel-dev/1panel/) - 现代化、开源的 Linux 服务器运维管理面板
- [MaxKB](https://github.com/1panel-dev/MaxKB/) - 强大易用的企业级智能体平台
- [JumpServer](https://github.com/jumpserver/jumpserver/) - 广受欢迎的开源堡垒机
- [Cordys CRM](https://github.com/1Panel-dev/CordysCRM) - 新一代的开源 AI CRM 系统
- [Halo](https://github.com/halo-dev/halo/) - 强大易用的开源建站工具
- [MeterSphere](https://github.com/metersphere/metersphere/) - 新一代的开源持续测试工具

## License

本仓库遵循 [FIT2CLOUD Open Source License](LICENSE) 开源协议，该许可证本质上是 GPLv3，但有一些额外的限制。

你可以基于 SQLBot 的源代码进行二次开发，但是需要遵守以下规定：

- 不能替换和修改 SQLBot 的 Logo 和版权信息；
- 二次开发后的衍生作品必须遵守 GPL V3 的开源义务。

如需商业授权，请联系 support@fit2cloud.com 。
