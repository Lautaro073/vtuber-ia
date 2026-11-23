# Modo NYRA para agy: apaga todos los plugins y servidores MCP (agy arranca y responde más rápido)
# y guarda cuáles estaban prendidos para volver a dejarlos exactamente igual.
#   Apagar:    powershell -File scripts\agy-nyra.ps1
#   Restaurar: powershell -File scripts\agy-nyra.ps1 -Restaurar
param([switch]$Restaurar)
$ErrorActionPreference = 'Stop'
$agy = Join-Path $env:LOCALAPPDATA 'agy\bin\agy.exe'
$estado = Join-Path $PSScriptRoot 'agy-estado.json'
if (-not (Test-Path $agy)) { Write-Host "No encuentro agy en $agy"; exit 1 }

if (-not $Restaurar) {
    if (Test-Path $estado) {
        Write-Host 'agy ya está en modo NYRA (hay un estado guardado). Si querés volver, usá el .bat de restaurar.'
        exit 1
    }
    $plugins = @((& $agy plugin list --json | Out-String | ConvertFrom-Json) | Where-Object { $_.enabled } | ForEach-Object { $_.name })
    # Tabla "NOMBRE TIPO ESTADO ...": solo se leen las tres primeras columnas (la cuarta puede traer claves)
    $mcps = @(& $agy mcp list | Select-Object -Skip 1 | ForEach-Object {
        $c = ($_.Trim() -split '\s+'); if ($c.Count -ge 3 -and $c[2] -eq 'enabled') { $c[0] } })
    [ordered]@{ fecha = (Get-Date).ToString('s'); plugins = $plugins; mcp = $mcps } |
        ConvertTo-Json | Set-Content -Path $estado -Encoding UTF8
    Write-Host "Guardado: $($plugins.Count) plugins y $($mcps.Count) MCP prendidos."
    foreach ($p in $plugins) { & $agy plugin disable $p | Out-Null; Write-Host "  plugin apagado: $p" }
    foreach ($m in $mcps) { & $agy mcp disable $m | Out-Null; Write-Host "  MCP apagado:    $m" }
    Write-Host ''
    Write-Host 'Listo: agy en modo NYRA. Reiniciá el server de NYRA para que tome el cambio.'
}
else {
    if (-not (Test-Path $estado)) { Write-Host 'No hay estado guardado: agy no estaba en modo NYRA.'; exit 1 }
    $e = Get-Content $estado -Raw | ConvertFrom-Json
    foreach ($p in @($e.plugins)) { if ($p) { & $agy plugin enable $p | Out-Null; Write-Host "  plugin prendido: $p" } }
    foreach ($m in @($e.mcp)) { if ($m) { & $agy mcp enable $m | Out-Null; Write-Host "  MCP prendido:    $m" } }
    Remove-Item $estado
    Write-Host ''
    Write-Host 'Listo: agy volvió a como estaba (todos sus plugins y MCP).'
}
