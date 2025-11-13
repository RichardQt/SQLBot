# 默认密码安全漏洞修复说明

## 漏洞描述

**漏洞名称**: 默认密码泄露  
**风险等级**: 高危  
**影响端点**: `GET /api/v1/user/defaultPwd`

### 漏洞详情

系统存在一个未授权的 API 端点 `/api/v1/user/defaultPwd`，任何人都可以通过该端点获取系统的默认密码 `SQLBot@123456`。这允许攻击者：

1. 获取系统默认密码
2. 尝试使用默认密码登录系统
3. 在管理员重置用户密码后的时间窗口内获取访问权限

## 修复方案

### 1. 后端修复

**文件**: `backend/apps/system/api/user.py`

- **删除**: 移除了暴露默认密码的 API 端点 `/api/v1/user/defaultPwd`
- **原因**: 默认密码是敏感信息，不应通过任何 API 端点暴露给客户端

```python
# 已删除的危险代码
# @router.get("/defaultPwd")
# async def default_pwd() -> str:
#     return settings.DEFAULT_PWD
```

### 2. 前端修复

**文件**:

- `frontend/src/api/user.ts` - 删除 API 调用
- `frontend/src/views/system/user/User.vue` - 移除密码显示功能
- `frontend/src/i18n/*.json` - 添加新的提示文本

**修改内容**:

- 移除了获取默认密码的 API 调用
- 移除了在重置密码弹窗中显示默认密码的功能
- 移除了复制默认密码的功能
- 更新用户界面提示文本，不再显示具体密码

### 3. 国际化文本更新

为重置密码确认对话框添加了新的提示文本：

- **中文**: "确认要将该用户的密码重置为默认密码吗？重置后，用户需要联系管理员获取新密码。"
- **英文**: "Are you sure you want to reset this user's password to the default password? After resetting, the user will need to contact the administrator to obtain the new password."
- **韩文**: "이 사용자의 비밀번호를 기본 비밀번호로 재설정하시겠습니까? 재설정 후 사용자는 관리자에게 문의하여 새 비밀번호를 받아야 합니다."

## 安全影响

### 修复前

- ❌ 任何人都可以获取默认密码
- ❌ 未授权访问风险高
- ❌ 密码明文显示在前端

### 修复后

- ✅ 默认密码仅在服务器端配置
- ✅ 前端无法获取默认密码
- ✅ 管理员通过其他安全渠道通知用户密码

## 后续建议

### 短期改进

1. ✅ **已完成**: 删除暴露默认密码的 API 端点
2. ⚠️ **建议**: 修改默认密码为更强的密码
3. ⚠️ **建议**: 在首次部署时强制管理员修改默认密码

### 长期改进

1. 📋 实施更安全的密码重置机制：
   - 生成随机密码而非固定默认密码
   - 通过加密邮件发送临时密码
   - 实施密码过期机制
2. 📋 添加安全审计：

   - 记录所有密码重置操作
   - 监控异常登录尝试
   - 实施账户锁定策略

3. 📋 增强密码策略：
   - 强制首次登录修改密码
   - 定期密码更新提醒
   - 密码历史记录防止重用

## 配置说明

默认密码仍在以下配置文件中配置（仅服务器端）：

- `backend/common/core/config.py` - `DEFAULT_PWD`
- `docker-compose.yaml` - 环境变量 `DEFAULT_PWD`
- `installer/install.conf` - `SQLBOT_DEFAULT_PWD`

**重要**: 建议在生产环境中修改这些配置文件中的默认密码为强密码。

## 测试验证

修复后应验证：

1. ✅ `/api/v1/user/defaultPwd` 端点返回 404 或不可访问
2. ✅ 管理员仍可以正常重置用户密码
3. ✅ 重置后的密码仍为配置文件中的默认密码
4. ✅ 前端不显示默认密码信息

## 修复日期

2025 年 11 月 13 日

## 修复人员

GitHub Copilot AI Assistant
