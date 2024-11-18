# Docker JupyterLab Launch Script
param(
    [string]$containerName = "tf_gpu",  # Change this to match your container name
    [string]$workDir = "C:/work/pressure_predict",          # Change this to your desired work directory
    [int]$jupyterPort = 8888,              # Change this to your desired port
    [switch]$interactive,                   # Switch for interactive mode
    [string]$mountDrive = "C"              # Drive letter to mount (default C:)
)
# $interactive = $true # debug mode

function Get-ContainerExists {
    param($name)
    $exists = wsl docker ps -a --filter "name=$name" --format '{{.Names}}'
    return $exists -eq $name
}

function Get-ContainerRunning {
    param($name)
    $status = wsl docker ps --filter "name=$name" --format '{{.Status}}'
    return $status -ne $null -and $status -ne ""
}

function Get-JupyterUrl {
    param(
        [string]$containerName,
        [int]$maxAttempts = 30
    )
    
    for ($i = 1; $i -le $maxAttempts; $i++) {
        Write-Host "Checking Jupyter status (attempt $i of $maxAttempts)..."
        $logs = wsl docker logs $containerName 2>&1
        
        if ($logs -match "http://[0-9\.]+:$jupyterPort/lab\?token=([^\s]+)") {
            return $matches[0]
        }
        
        Start-Sleep -Seconds 1
    }
    return $null
}

try {
    # Start WSL if not running
    Write-Host "Starting WSL..."
    wsl echo "WSL is running"

    # Check container state
    $containerExists = Get-ContainerExists $containerName
    $containerRunning = Get-ContainerRunning $containerName
    
    # Handle existing container
    if ($containerExists) {
        if ($containerRunning) {
            Write-Host "Container $containerName is running"
            if ($interactive) {
                Write-Host "Attaching to container in interactive mode..."
                wsl docker exec -it $containerName bash
                exit 0
            }
        } else {
            Write-Host "Removing stopped container $containerName..."
            wsl docker rm $containerName
        }
    }

    # Create docker run command
    $baseCommand = "docker run"
    $mountOptions = @(
        "--name `"$containerName`""
        "-p `"${jupyterPort}:${jupyterPort}`""
        "-v `"/$mountDrive/:/mnt/$mountDrive`""
        "-v `"${workDir}:/workspace`""
        "-e `"HOST_DRIVEPATH=/mnt/$mountDrive`""
    )

    if ($interactive) {
        Write-Host "Starting container in interactive mode..."
        $command = "$baseCommand -it --rm $($mountOptions -join ' ') tensorflow/tensorflow:latest-jupyter bash"
        Write-Host "Running command: $command"
        wsl $command
    }
    else {
        Write-Host "Starting container in Jupyter mode..."
        $command = "$baseCommand -d $($mountOptions -join ' ') tensorflow/tensorflow:latest-jupyter jupyter lab --notebook-dir=/workspace --ip 0.0.0.0 --port $jupyterPort --no-browser --allow-root"
        Write-Host "Running command: $command"
        $containerId = wsl $command

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start container. Exit code: $LASTEXITCODE"
        }

        Write-Host "Container started with ID: $containerId"
        
        # Wait for container to be running
        Start-Sleep -Seconds 2
        
        # Get Jupyter URL
        $jupyterUrl = Get-JupyterUrl -containerName $containerName
        
        if ($jupyterUrl) {
            Write-Host "Opening Jupyter in default browser..."
            Start-Process $jupyterUrl
            
            Write-Host "`nSetup complete! Jupyter Lab is running."
            Write-Host "Your $mountDrive drive is mounted at /mnt/$mountDrive inside the container"
            Write-Host "Your working directory is mounted at /workspace"
            Write-Host "`nUseful commands:"
            Write-Host "- To stop the container: wsl docker stop $containerName"
            Write-Host "- To enter the container: wsl docker exec -it $containerName bash"
            Write-Host "- To view container logs: wsl docker logs $containerName"
        }
        else {
            throw "Jupyter failed to start within the expected time"
        }
    }
}
catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    
    # Show container logs if available
    if (Get-ContainerExists $containerName) {
        Write-Host "`nContainer logs:" -ForegroundColor Yellow
        wsl docker logs $containerName 2>&1
        
        Write-Host "`nCleaning up container..." -ForegroundColor Yellow
        wsl docker stop $containerName 2>&1 | Out-Null
        wsl docker rm $containerName 2>&1 | Out-Null
    }
    
    exit 1
}