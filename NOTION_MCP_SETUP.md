# Notion MCP Server 配置指南

## 配置状态

✅ 已创建 `.vscode/mcp.json` 配置文件
✅ notion-mcp-server 已可用
⏳ 需要设置 Notion Integration Token

## 获取 Notion Integration Token 步骤

### 1. 创建 Notion Integration

1. 访问 [Notion Integrations](https://www.notion.so/my-integrations)
2. 点击 **"New integration"**
3. 填写集成名称（如："My MCP Integration"）
4. 选择需要关联的工作空间
5. 点击 **"Submit"** 创建集成

### 2. 获取 Token

创建成功后，复制 **"Internal Integration Token"**（格式为 `secret_...`）

### 3. 设置环境变量

在终端中运行：

```bash
# 临时设置（当前终端会话有效）
export NOTION_TOKEN="secret_你的token"

# 或者添加到 ~/.bashrc 永久生效
echo 'export NOTION_TOKEN="secret_你的token"' >> ~/.bashrc
source ~/.bashrc
```

### 4. 授权页面访问

1. 在 Notion 中打开需要访问的页面
2. 点击页面右上角的 **"..."** → **"Add connections"**
3. 选择你创建的 Integration

## 配置详情

当前配置文件 [`.vscode/mcp.json`](.vscode/mcp.json:1) 内容：

```json
{
  "servers": {
    "notion": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "NOTION_TOKEN": "${NOTION_TOKEN}"
      }
    }
  }
}
```

## 可用工具

notion-mcp-server 提供以下能力：

- **搜索页面**：`notion_search_pages`
- **获取页面**：`notion_get_page`
- **创建页面**：`notion_create_page`
- **更新页面**：`notion_update_page`
- **获取数据库**：`notion_get_database`
- **查询数据库**：`notion_query_database`
- **创建数据库条目**：`notion_create_database_item`
- **添加块到页面**：`notion_append_block_children`

## 测试连接

设置 Token 后，可以在对话中测试：

```
帮我列出我的 Notion 工作空间中的所有页面
```

或

```
在 Notion 中搜索关于 "项目" 的页面
```

## 参考链接

- [Notion MCP Server GitHub](https://github.com/makenotion/notion-mcp-server)
- [Notion API 文档](https://developers.notion.com/)
