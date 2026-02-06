# Aegis-Isle 快速启动脚本
# 用于 SillyTavern 真实测试

Write-Host "🚀 启动 Aegis-Isle 服务器..." -ForegroundColor Cyan
Write-Host ""

# 检查虚拟环境
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ 激活虚拟环境..." -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  未找到虚拟环境,使用全局 Python" -ForegroundColor Yellow
}

# 显示配置信息
Write-Host ""
Write-Host "📋 服务器配置:" -ForegroundColor Cyan
Write-Host "  - 地址: http://localhost:8000" -ForegroundColor White
Write-Host "  - API 文档: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - 状态目录: data/state/" -ForegroundColor White
Write-Host "  - 快照目录: data/snapshots/" -ForegroundColor White
Write-Host ""

# 显示 SillyTavern 配置
Write-Host "⚙️  SillyTavern 配置:" -ForegroundColor Cyan
Write-Host "  - API Type: OpenAI" -ForegroundColor White
Write-Host "  - API URL: http://localhost:8000/v1" -ForegroundColor White
Write-Host "  - Model: gpt-4 (任意)" -ForegroundColor White
Write-Host ""

# 启动服务器
Write-Host "🔥 启动中..." -ForegroundColor Green
Write-Host ""
python -m aegis_isle.api.server

# 如果服务器退出
Write-Host ""
Write-Host "❌ 服务器已停止" -ForegroundColor Red
