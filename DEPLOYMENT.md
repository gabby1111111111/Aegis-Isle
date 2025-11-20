# AegisIsle 部署指南

## 概述

AegisIsle 是一个多智能体协作 RAG 系统，支持完整的 OAuth2 + RBAC 权限控制和结构化审计日志。本指南详细说明了如何在开发和生产环境中部署该系统。

## 系统架构

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Frontend      │   Backend       │   Infrastructure│
│                 │                 │                 │
│ • Web UI        │ • FastAPI       │ • Docker        │
│ • Admin Panel   │ • OAuth2/RBAC   │ • Redis         │
│                 │ • Audit Logs    │ • Qdrant        │
│                 │ • RAG Pipeline  │ • PostgreSQL    │
│                 │ • Agent System  │ • ELK Stack     │
└─────────────────┴─────────────────┴─────────────────┘
```

## 先决条件

### 系统要求

- **操作系统**: Ubuntu 22.04+ / CentOS 8+ / macOS 12+ / Windows 10+
- **内存**: 最小 8GB RAM (推荐 16GB+)
- **存储**: 最小 50GB 可用空间
- **CPU**: 4+ 核心 (推荐 8+ 核心)
- **网络**: 稳定的网络连接用于模型下载

### 软件依赖

- **Docker**: 20.10+ 和 Docker Compose v2.0+
- **Python**: 3.11+ (如果本地开发)
- **Git**: 最新版本
- **Node.js**: 18+ (如果需要前端开发)

## 快速开始 (Docker 部署)

### 1. 克隆项目

```bash
git clone https://github.com/your-org/aegis-isle.git
cd aegis-isle
```

### 2. 环境配置

复制并编辑环境配置文件:

```bash
cp .env.example .env
```

编辑 `.env` 文件中的关键配置:

```bash
# 基础配置
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=your-super-secret-key-here-32-chars-min

# 数据库配置
DATABASE_URL=postgresql://user:password@postgres:5432/aegis_isle
REDIS_URL=redis://redis:6379/0
QDRANT_HOST=qdrant

# AI 模型 API 密钥
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# 认证配置
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure-admin-password-123!
```

### 3. 启动服务

使用 Docker Compose 启动所有服务:

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

### 4. 验证部署

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 获取认证 token
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secure-admin-password-123!"

# 测试管理员权限
curl -X GET "http://localhost:8000/api/v1/admin/config" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 生产环境部署

### 1. 服务器准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 重启以应用用户组更改
sudo reboot
```

### 2. 生产环境配置

创建生产环境配置目录:

```bash
sudo mkdir -p /opt/aegis-isle/{config,data,logs}
sudo chown -R $USER:$USER /opt/aegis-isle
cd /opt/aegis-isle
git clone https://github.com/your-org/aegis-isle.git .
```

配置生产环境变量:

```bash
# 创建生产环境配置
cat > .env.production << EOF
# 生产环境配置
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# 安全配置
SECRET_KEY=$(openssl rand -hex 32)
ADMIN_USERNAME=aegis_admin
ADMIN_PASSWORD=$(openssl rand -base64 32)

# 数据库配置
DATABASE_URL=postgresql://aegis_user:$(openssl rand -base64 16)@postgres:5432/aegis_isle
REDIS_URL=redis://:$(openssl rand -base64 16)@redis:6379/0
QDRANT_HOST=qdrant

# AI 配置 (需要替换为实际密钥)
OPENAI_API_KEY=sk-your-production-openai-key
ANTHROPIC_API_KEY=your-production-anthropic-key

# 网络配置
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_HOSTS=your-domain.com,*.your-domain.com

# 日志和监控
LOG_REQUESTS=True
ENABLE_METRICS=True
AUDIT_LOG_ENABLED=True

# 资源限制
MAX_TOKENS=4096
CHUNK_SIZE=1000
MAX_RETRIEVED_DOCS=10
EOF
```

### 3. SSL/TLS 配置 (使用 Nginx)

创建 Nginx 配置:

```bash
sudo mkdir -p /opt/aegis-isle/nginx

cat > /opt/aegis-isle/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # 重定向到 HTTPS
        return 301 https://\$server_name\$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/certs/aegis-isle.crt;
        ssl_certificate_key /etc/ssl/private/aegis-isle.key;

        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        client_max_body_size 50M;

        location / {
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;

            # WebSocket 支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /docs {
            return 404;  # 生产环境中禁用 API 文档
        }
    }
}
EOF
```

更新 `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - backend
    restart: unless-stopped

  backend:
    build: .
    env_file: .env.production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
        reservations:
          memory: 2G
          cpus: "1.0"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: aegis_isle
      POSTGRES_USER: aegis_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aegis_user -d aegis_isle"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass_file /run/secrets/redis_password
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "$(cat /run/secrets/redis_password)", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:v1.7.0
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  redis_password:
    file: ./secrets/redis_password.txt

volumes:
  postgres_data:
  redis_data:
  qdrant_data:

networks:
  default:
    driver: bridge
```

### 4. 生产环境启动

```bash
# 创建密钥文件
mkdir -p secrets
echo "$(openssl rand -base64 32)" > secrets/postgres_password.txt
echo "$(openssl rand -base64 32)" > secrets/redis_password.txt

# 设置适当的权限
chmod 600 secrets/*

# 启动生产服务
docker-compose -f docker-compose.prod.yml up -d

# 查看服务状态
docker-compose -f docker-compose.prod.yml ps
```

## 监控和日志管理

### 1. ELK Stack 部署

创建 ELK 配置:

```yaml
# elk-stack.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    restart: unless-stopped

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs/audit:/var/log/aegis-isle/audit
    environment:
      - xpack.monitoring.elasticsearch.hosts=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped

volumes:
  elasticsearch_data:
```

Logstash 配置文件 (`logstash/pipeline/logstash.conf`):

```ruby
input {
  file {
    path => "/var/log/aegis-isle/audit/*.jsonl"
    start_position => "beginning"
    codec => "json"
    type => "audit"
  }
}

filter {
  if [type] == "audit" {
    date {
      match => [ "@timestamp", "ISO8601" ]
    }

    if [event_type] == "authentication" {
      mutate {
        add_tag => [ "security", "auth" ]
      }
    }

    if [event_type] == "authorization" {
      mutate {
        add_tag => [ "security", "authz" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "aegis-isle-audit-%{+YYYY.MM.dd}"
  }
}
```

### 2. Prometheus 监控

创建 `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aegis-isle'
    static_configs:
      - targets: ['backend:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

### 3. Grafana 仪表板

```yaml
# monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

## 安全配置

### 1. 防火墙配置

```bash
# Ubuntu/Debian 使用 UFW
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 限制管理端口访问
sudo ufw allow from 10.0.0.0/8 to any port 3000  # Grafana
sudo ufw allow from 10.0.0.0/8 to any port 5601  # Kibana
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus
```

### 2. SSL 证书配置

使用 Let's Encrypt:

```bash
# 安装 Certbot
sudo apt install certbot

# 获取证书
sudo certbot certonly --standalone -d your-domain.com

# 设置自动续期
sudo crontab -e
# 添加: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. 备份配置

创建备份脚本 (`backup.sh`):

```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/aegis-isle"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份数据库
docker-compose exec postgres pg_dump -U aegis_user aegis_isle > "$BACKUP_DIR/postgres_$DATE.sql"

# 备份 Redis
docker-compose exec redis redis-cli --rdb /data/dump_$DATE.rdb

# 备份 Qdrant
docker run --rm -v qdrant_data:/data -v "$BACKUP_DIR":/backup busybox tar czf /backup/qdrant_$DATE.tar.gz -C /data .

# 备份配置文件
tar czf "$BACKUP_DIR/config_$DATE.tar.gz" .env* docker-compose*.yml secrets/ nginx/

# 清理旧备份 (保留 30 天)
find "$BACKUP_DIR" -mtime +30 -delete

echo "备份完成: $DATE"
```

设置定期备份:

```bash
chmod +x backup.sh
sudo crontab -e
# 添加: 0 2 * * * /opt/aegis-isle/backup.sh
```

## 故障排除

### 常见问题

#### 1. 容器无法启动

```bash
# 查看容器日志
docker-compose logs backend

# 检查资源使用
docker stats

# 重启服务
docker-compose restart backend
```

#### 2. 数据库连接问题

```bash
# 检查数据库连接
docker-compose exec backend python -c "from src.aegis_isle.core.config import settings; print(settings.database_url)"

# 测试数据库连接
docker-compose exec postgres psql -U aegis_user -d aegis_isle -c "SELECT 1;"
```

#### 3. 认证问题

```bash
# 检查管理员用户
docker-compose exec backend python -c "
from src.aegis_isle.api.dependencies import USERS_DB
print(list(USERS_DB.keys()))
"

# 重置管理员密码
docker-compose exec backend python -c "
from src.aegis_isle.api.dependencies import USERS_DB, pwd_context
USERS_DB['admin'].hashed_password = pwd_context.hash('newpassword123')
print('密码已重置')
"
```

### 性能优化

#### 1. 数据库优化

```sql
-- PostgreSQL 性能调优
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

#### 2. 应用优化

```bash
# 增加 worker 进程数
# 在 Dockerfile 中设置:
CMD ["uvicorn", "src.aegis_isle.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## API 使用指南

### 认证流程

```bash
# 1. 获取访问令牌
curl -X POST "https://your-domain.com/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your-password"

# 响应:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user_info": {
    "user_id": "admin_001",
    "username": "admin",
    "roles": ["user", "admin", "super_admin"]
  }
}

# 2. 使用令牌访问受保护的端点
curl -X GET "https://your-domain.com/api/v1/admin/config" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 主要 API 端点

| 端点 | 方法 | 权限 | 描述 |
|------|------|------|------|
| `/api/v1/auth/token` | POST | 公开 | 获取访问令牌 |
| `/api/v1/auth/me` | GET | 用户 | 获取当前用户信息 |
| `/api/v1/documents/upload` | POST | 用户 | 上传文档 |
| `/api/v1/query` | POST | 用户 | 执行 RAG 查询 |
| `/api/v1/agents/execute` | POST | 用户 | 执行智能体任务 |
| `/api/v1/admin/config` | GET | 管理员 | 获取系统配置 |
| `/api/v1/admin/stats` | GET | 管理员 | 获取系统统计 |

## 更新和维护

### 1. 应用更新

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build --no-cache backend

# 滚动更新
docker-compose up -d backend
```

### 2. 数据库迁移

```bash
# 运行数据库迁移
docker-compose exec backend python -m alembic upgrade head
```

### 3. 健康检查脚本

创建 `health_check.sh`:

```bash
#!/bin/bash

# 检查服务健康状态
services=("backend" "postgres" "redis" "qdrant")

for service in "${services[@]}"; do
    status=$(docker-compose ps -q $service | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null)
    if [ "$status" != "healthy" ]; then
        echo "WARNING: $service is not healthy (status: $status)"
        # 发送告警通知
        curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
          -H 'Content-type: application/json' \
          --data "{\"text\":\"AegisIsle 服务异常: $service 状态 $status\"}"
    else
        echo "OK: $service is healthy"
    fi
done
```

### 4. 自动监控

使用 systemd 创建服务监控:

```ini
# /etc/systemd/system/aegis-isle-monitor.service
[Unit]
Description=AegisIsle Health Monitor
After=docker.service

[Service]
Type=oneshot
User=aegis
ExecStart=/opt/aegis-isle/health_check.sh

# /etc/systemd/system/aegis-isle-monitor.timer
[Unit]
Description=Run AegisIsle Health Monitor every 5 minutes
Requires=aegis-isle-monitor.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
```

启用监控:

```bash
sudo systemctl enable aegis-isle-monitor.timer
sudo systemctl start aegis-isle-monitor.timer
```

## 支持和联系

如遇问题，请按以下顺序处理：

1. 查看本文档的故障排除部分
2. 检查项目 Issues: https://github.com/your-org/aegis-isle/issues
3. 查看系统日志和审计日志
4. 联系技术支持团队

---

**注意**: 本指南假设您有基本的 Docker、Linux 系统管理和网络配置经验。在生产环境中部署前，请确保您已经充分测试了所有配置。