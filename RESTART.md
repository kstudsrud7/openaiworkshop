Perfect — here’s a clean PowerShell restart flow for all 3 services.

- Stop old listeners (safe cleanup):
  - `7000,8000,3000 | ForEach-Object { $c=Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){ Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } }`
- Terminal 1 (MCP):
  - `cd C:\Users\Admin\repos\OpenAIWorkshop\mcp`
  - `uv run python mcp_service.py`
- Terminal 2 (Backend):
  - `cd C:\Users\Admin\repos\OpenAIWorkshop\agentic_ai\applications`
  - `uv run python backend.py`
- Terminal 3 (Frontend React):
  - `cd C:\Users\Admin\repos\OpenAIWorkshop\agentic_ai\applications\react-frontend`
  - `npm run dev`

Config change rule: restart only the service whose .env/config changed (backend for backend .env, MCP for MCP config, frontend for Vite env). You do **not** need to restart your whole localhost/PC.