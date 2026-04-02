# SQLBot现状分析报告

## 目录

1. [项目概述](#1-项目概述)
2. [数据库设计](#2-数据库设计)
3. [测试场景与准确率](#3-测试场景与准确率)
4. [后续优化领域](#4-后续优化领域)

## 1. 项目概述

SQLBot是一个**Text2SQL智能查询系统**，旨在将自然语言转换为SQL查询语句，帮助用户通过日常语言与数据库进行交互，无需掌握复杂的SQL语法。

**核心目标：**

- 降低数据库查询门槛
- 提高数据查询效率
- 实现自然语言到SQL的准确转换

## 2. 数据库设计

**注：仅列出目前所涉及的表结构**

### 2.1 fx_article_records（法宣所有文章记录表）

**表说明：** 存储所有法宣文章的详细记录信息

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id              | bigint   | 自增主键                       |
| crawl_time      | datetime | 爬取时间                       |
| crawl_channel   | varchar  | 爬取渠道（官网、微信公众号等） |
| article_title   | varchar  | 文章标题                       |
| article_content | text     | 文章内容                       |
| publish_time    | datetime | 文章发布时间                   |
| view_count      | int      | 浏览次数                       |

### 2.2 fx_category_labels（分类标签表）

**表说明：** 管理文章分类标签信息

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id               | bigint  | 自增主键                                       |
| unit_property    | varchar | 单位属性（市级政府、区级政府、高校、市属国企） |
| industry_system  | varchar | 行业系统（纪委监委系统、教育系统等）           |
| law_content_type | varchar | 普法内容类型（宪法、民法典等）                 |
| staff_size       | varchar | 人数规模（1-50人、51-150人等）                 |
| target_audience  | varchar | 受众群体（青少年、居民等） |
| remark | text | 备注 |
| creator | varchar | 创建人 |

### 2.3 fx_data_unit（部门表）

**表说明：** 存储各部门/单位的基本信息

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id        | bigint  | 自增id                          |
| parent_id | bigint  | 父id                            |
| unit_name | varchar | 单位属性名称或单位名称          |
| order_num | int     | 显示顺序                        |
| status    | char    | 使用状态（0正常 1停用）         |
| del_flag  | char    | 删除标志（0代表存在 2代表删除） |
| create_by | varchar | 创建者                          |

### 2.4 fx_education_articles（普法教育文章信息表）

**表说明：** 专门存储普法教育相关文章

| 字段名 | 类型 | 注释 |
|--------|------|------|
| LLM_id          | bigint  | 自增主键                         |
| id              | bigint  | 记录表的id                       |
| article_id      | varchar | 文章ID                           |
| unit_name       | varchar | 单位名称                         |
| unit_property   | varchar | 单位属性（市级单位、区级单位等） |
| industry_system | varchar | 行业系统（法院系统、检察系统等） |
| unit_district   | varchar | 单位所在区                       |
| people_scale    | varchar | 人数规模（1-50人、51-100人等）） |
| month           | char    | 月份（格式：YYYY-MM）            |

### 2.5 fx_education_articles_group（普法教育文章受众群体表）

**表说明：** 记录普法文章的受众群体关联关系

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id_group     | bigint  | 自增主键                      |
| article_id   | varchar | 文章ID                        |
| target_group | varchar | 受众群体 (居民、学生、企业等) |

### 2.6 fx_education_articles_legal（普法教育文章法律内容表）

**表说明：** 存储普法文章涉及的法律内容

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id_legal           | bigint  | 自增主键                |
| article_id         | varchar | 文章ID                  |
| legal_content_type | varchar | 普法内容类型 (RAB, BAS) |
| Legal_topics       | varchar | 法律主题字段名          |

### 2.7 fx_monthly_content_data（月度内容数据表）

**表说明：** 按月统计的内容数据汇总

| 字段名 | 类型 | 注释 |
|--------|------|------|
| time_period   | varchar | 年月             |
| content_key   | varchar | 段落名           |
| content_value | text    | 段落内容         |
| is_json       | tinyint | 是否为json       |
| position      | int     | 段落在文章的顺序 |

### 2.8 fx_theme（法律主题月表）

**表说明：** 管理法律主题月活动信息

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id          | bigint    | 自增主键                                 |
| year        | int       | 年份                                     |
| theme_name  | varchar   | 法律主题月名称                           |
| start_date  | date      | 开始时间                                 |
| end_date    | date      | 结束时间                                 |
| modifier    | varchar   | 修改人                                   |
| modify_time | timestamp | 修改时间                                 |
| status | tinyint | 状态：0-废弃，1-使用 |
| generate | tinyint | 状态：0-主题报告未生成，1-主题报告已生成 |

### 2.9 fx_theme_law_relation（法律主题与法律法规关联表）

**表说明：** 关联法律主题与相关法律法规

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id         | bigint  | 自增主键                                    |
| year       | int     | 年份                                        |
| theme_name | varchar | 主题名称（对应 fx_theme_yuebao.theme_name） |
| law_name   | varchar | 法律法规名称                                |
| modifier   | varchar | 修改人                                      |

### 2.10 fx_theme_report（主题报告表）

**表说明：** 存储各主题活动的报告数据

| 字段名 | 类型 | 注释 |
|--------|------|------|
| id          | bigint    | 自增主键                                 |
| year        | int       | 年份                                     |
| theme_name  | varchar   | 法律主题月名称                           |
| start_date  | date      | 开始时间                                 |
| end_date    | date      | 结束时间                                 |
| modifier    | varchar   | 修改人                                   |
| modify_time | timestamp | 修改时间                                 |
| status      | tinyint   | 状态：0-废弃，1-使用                     |
| generate    | tinyint   | 状态：0-主题报告未生成，1-主题报告已生成 |

### 2.11 fx_unit_gzh_contrast（单位与公众号对比表）

**表说明：** 对比各单位公众号运营数据

| 字段名 | 类型 | 注释 |
|--------|------|------|
| unit_name | varchar | 单位名称   |
| gzh_name  | varchar | 公众号名称 |

## 3. 测试场景与准确率

### 3.1 总体准确率

**系统整体准确率：92% （不统计简单领域）**

### 3.2 测试领域1：法律主题与受众群体查询

**测试问题：**

1. 查询2025年发布的关于"民法典"的普法教育文章标题
2. 找出针对受众群体为"青少年"群体的前五篇文章
3. 使用柱状图展示2025年普法文章的法律主题分布
4. 查询包含"劳动法"条款的文章标题和来源单位
5. 展示2025年普法文章的法律主题为：世界水日及其单位名称
6. ...

**准确率：90%  18/20**

### 3.3 测试领域2：单位文章数量统计与排序

**测试问题：**

1. 统计2025年9月各单位发布的文章总数 
2. 统计每个单位发布的普法文章数量
3. 查询阅读量总数超过10000的单位
4. 找出阅读量最高的5篇普法文章
5. 查询2025年普法文章中普法内容类型为刑法的文章数量
6. 对比各单位公众号的阅读量和月发布文章数
7. ...

**准确率：95%  19/20** 

### 3.4 测试领域3：具体文章点赞量、阅读量等分析

1、‘央视《新闻联播》报道！东大人这样说……’ 这篇文章的点赞量是多少以及其所发布的单位名称

2、东大放假啦！“最全版”寒假生活指南请查收！这篇文章的阅读量是多少以及其所发布的单位名称

3、....

**准确率：99%  10/10**

### 3.5 部分单轮测试失败示例问题（第二轮测试均成功）

1、使用折线图展示2025年普法文章的分享量趋势

2、使用饼图展示2025年普法文章的单位属性分布

3、使用折线图展示2025年普法文章的浏览次数趋势

4、使用饼图展示东南大学2025年普法文章的人数规模占比

5、...

(第二轮测试大部分成功---问题原因：大模型输出不稳定)

## 4.后续优化领域

- 针对法宣领域进一步优化，包含但不限于机构信息、法律法规、受众群体、活动规模等术语配置和sql示例添加，同时针对系统提示词部分加上针对法宣领域的特定提示词限制