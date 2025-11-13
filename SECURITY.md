# SQLBot 安全配置指南

## 🔐 重要安全警告

在生产环境部署 SQLBot 之前，**必须**完成以下安全配置，否则系统将面临严重的安全风险。

## 一、JWT 密钥配置（必须）

### 1.1 问题说明

JWT (JSON Web Token) 密钥 `SECRET_KEY` 用于签名和验证所有的访问令牌。如果使用默认或公开的密钥，攻击者可以：

- 伪造任意用户的身份令牌
- 绕过登录验证直接访问系统
- 获取管理员权限
- 访问和修改敏感数据

### 1.2 配置步骤

#### 步骤 1: 生成强随机密钥

在服务器上运行以下命令生成一个安全的随机密钥：

```bash
# Linux/Mac
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Windows PowerShell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

这将生成一个类似于以下的密钥：

```text
xK8dP2mN5wQ7rT9yU3vB6nC8eF1hG4jL0oM3pR6sV9zW2aD5fH8kN1qT4xY7
```

#### 步骤 2: 更新 .env 文件

编辑项目根目录下的 `.env` 文件，将 `SECRET_KEY` 的值替换为刚才生成的密钥：

```bash
# 找到这一行
SECRET_KEY=PLEASE_CHANGE_THIS_TO_A_RANDOM_SECRET_KEY

# 替换为生成的密钥
SECRET_KEY=xK8dP2mN5wQ7rT9yU3vB6nC8eF1hG4jL0oM3pR6sV9zW2aD5fH8kN1qT4xY7
```

#### 步骤 3: 保护密钥安全

- ✅ **必须做**

  - 每个部署环境（开发、测试、生产）使用不同的密钥
  - 将 `.env` 文件添加到 `.gitignore`，防止提交到版本控制系统
  - 限制服务器上 `.env` 文件的访问权限（chmod 600）
  - 定期更换密钥（更换后所有用户需要重新登录）

- ❌ **禁止做**
  - 不要在代码中硬编码密钥
  - 不要将密钥提交到 Git 仓库
  - 不要在多个系统间共享相同的密钥
  - 不要在日志或错误信息中输出密钥

### 1.3 验证配置

重启应用后，尝试使用旧的 token 访问 API，应该会收到 "Invalid token signature" 错误，这表示配置生效。

## 二、数据库密码配置

### 2.1 修改默认密码

默认的数据库密码是 `Password123@pg`，在生产环境中必须修改：

```bash
# PostgreSQL 密码配置
POSTGRES_PASSWORD=你的强密码
```

### 2.2 密码要求

- 至少 12 个字符
- 包含大小写字母、数字和特殊字符
- 不使用常见单词或字典词汇

## 三、管理员密码配置

### 3.1 修改默认管理员密码

首次安装后，立即登录并修改默认密码 `SQLBot@123456`。

也可以在 `.env` 中修改默认密码：

```bash
DEFAULT_PWD="你的强密码"
```

### 3.2 密码策略建议

- 强制要求所有用户使用强密码
- 定期要求用户更换密码
- 启用账户锁定机制（多次登录失败后锁定）

## 四、CORS 配置

### 4.1 限制跨域访问

在 `.env` 中配置允许的前端域名：

```bash
BACKEND_CORS_ORIGINS="http://localhost:5173,https://your-domain.com"
```

⚠️ 生产环境中不要使用 `*` 允许所有域名访问。

## 五、HTTPS 配置

### 5.1 生产环境必须使用 HTTPS

配置 Nginx 或其他反向代理来启用 HTTPS：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 六、安全检查清单

部署前请确认以下所有项目：

- [ ] 已生成并配置唯一的 `SECRET_KEY`
- [ ] 已修改数据库默认密码
- [ ] 已修改管理员默认密码
- [ ] 已配置正确的 CORS 策略
- [ ] 已启用 HTTPS（生产环境）
- [ ] 已将 `.env` 添加到 `.gitignore`
- [ ] 已限制 `.env` 文件的访问权限
- [ ] 已配置防火墙，仅开放必要端口
- [ ] 已启用日志记录和监控
- [ ] 已制定备份和恢复计划

## 七、安全事件响应

### 7.1 如果怀疑密钥泄露

1. 立即生成新的 `SECRET_KEY`
2. 更新 `.env` 文件
3. 重启应用
4. 通知所有用户重新登录
5. 检查日志，查看是否有异常访问
6. 考虑重置所有用户密码

### 7.2 联系方式

如发现安全漏洞，请通过以下方式报告：

- 邮件：`security@your-domain.com`
- 不要在公开 issue 中讨论安全漏洞

## 八、附加安全建议

### 8.1 日志监控

启用详细的安全日志：

```bash
LOG_LEVEL="INFO"  # 或 "DEBUG" 用于调试
```

定期检查日志中的异常模式：

- 多次失败的登录尝试
- 来自异常 IP 的访问
- 无效的 token 签名（可能是攻击尝试）

### 8.2 网络安全

- 使用防火墙限制入站连接
- 配置 fail2ban 防止暴力破解
- 使用 VPN 或 IP 白名单限制管理后台访问
- 定期更新系统和依赖包

### 8.3 数据备份

- 定期备份数据库
- 加密备份文件
- 测试恢复流程
- 将备份存储在安全的异地位置

## 九、合规性考虑

根据您的行业和地区，可能需要遵守特定的安全标准：

- GDPR（欧盟数据保护）
- HIPAA（美国医疗数据）
- PCI DSS（支付卡数据）
- 等保 2.0（中国信息安全）

请确保您的部署符合相关法规要求。

---

**最后更新**: 2025 年 11 月 13 日

**版本**: v2.1 - 加强了 JWT 安全机制和默认密钥检测

## 十、最新安全加固说明（v2.1）

### 10.1 修复的安全漏洞

本次更新修复了以下严重安全漏洞：

#### 1. 默认密钥问题（高危）

- **问题**: 系统使用了硬编码的默认 SECRET_KEY，攻击者可利用此密钥伪造 JWT 令牌
- **修复**: 移除所有硬编码密钥，安装时自动生成随机密钥，启动时强制检查密钥安全性

#### 2. JWT 签名验证问题（高危）

- **问题**: 部分验证方法缺少严格的签名验证
- **修复**: 强制启用签名验证，添加签发者验证，增加 JWT ID 防止重放攻击

#### 3. 令牌有效期过长（中危）

- **问题**: 访问令牌有效期为 8 天，风险较高
- **修复**: 缩短令牌有效期至 24 小时，添加签发时间字段

### 10.2 新增安全机制

系统启动时会自动执行安全检查：

- 检测已知弱密钥和默认密钥
- 验证密钥长度和复杂度
- 如检测到不安全密钥，系统拒绝启动

### 10.3 迁移指南

如使用旧版本，请立即：

1. 生成新密钥: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
2. 更新 `.env`、`docker-compose.yaml` 等配置文件中的 SECRET_KEY
3. 重启应用
4. 通知所有用户重新登录
