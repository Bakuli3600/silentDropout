$ErrorActionPreference = "Stop"

# Check backend
try {
    $resp = Invoke-RestMethod -Uri http://127.0.0.1:8800/api/health -Method Get -TimeoutSec 10
    Write-Output "Backend health: OK"
    $json = $resp | ConvertTo-Json -Depth 3
    Write-Output $json
} catch {
    Write-Output "Backend health: FAILED - $_"
}

Write-Output "---"

# Check frontend
try {
    $resp2 = Invoke-RestMethod -Uri http://localhost:3000 -Method Get -TimeoutSec 15
    Write-Output "Frontend HTML length: $($resp2.Length)"
} catch {
    Write-Output "Frontend: FAILED - $_"
}
