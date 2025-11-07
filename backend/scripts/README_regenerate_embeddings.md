# 重新生成 Embedding 脚本使用指南

## 问题背景

当出现以下错误时，说明数据库中存储的 embedding 向量维度与当前使用的模型维度不匹配:

```
ValueError: The vector dimension must be the same
```

### 常见原因

1. **更换了 embedding 模型**

   - 从 `text2vec-base-chinese` (768 维) 切换到 `bge-large-zh-v1.5` (1024 维)
   - 或反向切换

2. **模型配置不一致**
   - 不同环境使用了不同的模型
   - 数据是用旧模型生成的，但当前使用新模型

## 解决方案

### 方案 1: 使用重新生成脚本 (推荐)

#### Docker 环境

```bash
# 进入容器
docker exec -it sqlbot-backend bash

# 切换到 backend 目录
cd /opt/sqlbot/app

# 运行脚本
python scripts/regenerate_embeddings.py
```

#### 本地开发环境

```bash
# 进入 backend 目录
cd backend

# 激活虚拟环境 (如果使用)
# source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# 运行脚本
python scripts/regenerate_embeddings.py
```

### 脚本功能

1. **清空所有现有 embedding**

   - 清空所有表的 embedding 字段
   - 清空所有数据源的 embedding 字段

2. **重新生成所有 embedding**

   - 使用当前配置的模型重新生成所有表的 embedding
   - 使用当前配置的模型重新生成所有数据源的 embedding

3. **安全确认**
   - 清空数据后会提示用户确认是否继续
   - 支持取消操作

### 方案 2: 手动清空数据库

如果脚本无法运行，可以手动执行 SQL:

```sql
-- 清空表的 embedding
UPDATE core_table SET embedding = NULL;

-- 清空数据源的 embedding
UPDATE core_datasource SET embedding = NULL;
```

然后重启应用，系统会自动生成缺失的 embedding。

## 代码改进

为了避免未来出现类似问题，已在代码中添加了以下改进:

### 1. 维度检测

在 `calc_table_embedding` 和 `get_ds_embedding` 中添加了维度检测:

```python
# 检查向量维度是否匹配
if stored_dim != q_embedding_dim:
    SQLBotLogUtil.warning(
        f"embedding 维度不匹配: 存储={stored_dim}, 当前模型={q_embedding_dim}"
    )
    # 维度不匹配时设置相似度为 0
    _list[index]['cosine_similarity'] = 0.0
    continue
```

### 2. 错误处理

添加了更详细的错误日志，便于排查问题:

```python
try:
    stored_embedding = json.loads(item)
    _list[index]['cosine_similarity'] = cosine_similarity(q_embedding, stored_embedding)
except (json.JSONDecodeError, ValueError) as e:
    SQLBotLogUtil.warning(f"embedding 数据异常: {str(e)}")
    _list[index]['cosine_similarity'] = 0.0
```

## 预防措施

### 1. 确保模型一致性

在 `backend/common/core/config.py` 中检查配置:

```python
DEFAULT_EMBEDDING_MODEL: str = 'backend/bge-large-zh-v1.5'  # 确保这个配置是正确的
```

### 2. 模型切换流程

如果需要切换 embedding 模型:

1. 修改配置文件中的 `DEFAULT_EMBEDDING_MODEL`
2. 运行 `regenerate_embeddings.py` 脚本
3. 重启应用

### 3. 定期检查

可以在日志中查找警告信息:

```bash
# 查找维度不匹配的警告
grep "embedding 维度不匹配" logs/debug.log
```

## 注意事项

1. **耗时**: 根据数据量大小，重新生成可能需要几分钟到几十分钟
2. **资源**: embedding 生成会占用较多 CPU/GPU 资源
3. **备份**: 建议在操作前备份数据库
4. **并发**: 生成期间应用可以继续运行，但会看到维度不匹配的警告

## 模型维度对照表

| 模型名称              | 维度 | 路径                          |
| --------------------- | ---- | ----------------------------- |
| bge-large-zh-v1.5     | 1024 | backend/bge-large-zh-v1.5     |
| text2vec-base-chinese | 768  | backend/text2vec-base-chinese |

## 常见问题

### Q: 脚本运行后仍然报错？

A: 确认以下几点:

- 脚本是否成功完成
- 是否重启了应用
- 检查日志确认 embedding 确实被重新生成

### Q: 可以跳过某些表或数据源吗？

A: 可以修改脚本，添加过滤条件跳过特定的表或数据源

### Q: 生成过程可以中断吗？

A: 可以，但已生成的 embedding 会保留，未生成的下次启动时会自动生成

### Q: 影响现有功能吗？

A: 不会，系统会优雅降级。维度不匹配的记录相似度会被设为 0，不影响其他记录的使用
