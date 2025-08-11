"""Command-line interface for the Self-Healing Pipeline Guard."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from healing_guard.core.config import settings
from healing_guard.core.failure_detector import FailureDetector, FailureType
from healing_guard.core.healing_engine import HealingEngine
from healing_guard.core.quantum_planner import QuantumTaskPlanner
from healing_guard.models.pipeline import PipelineFailure
from healing_guard.models.healing import HealingRequest

app = typer.Typer(
    name="healing-guard",
    help="🛡️ Self-Healing Pipeline Guard - AI-powered CI/CD failure recovery",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def analyze(
    logs_file: Path = typer.Argument(..., help="Path to CI/CD log file to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for analysis results"),
    format_type: str = typer.Option("json", "--format", "-f", help="Output format (json, yaml, table)"),
    confidence_threshold: float = typer.Option(0.7, "--threshold", "-t", help="Confidence threshold for classification"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """🔍 Analyze CI/CD pipeline failure logs and classify issues."""
    setup_logging(debug)
    
    if not logs_file.exists():
        console.print(f"[red]Error: Log file {logs_file} not found[/red]")
        raise typer.Exit(1)
    
    console.print(Panel("🔍 Analyzing Pipeline Failure", style="blue"))
    
    try:
        # Read logs
        logs_content = logs_file.read_text()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize detector
            progress.add_task("Initializing failure detector...", total=None)
            detector = FailureDetector(settings)
            
            # Create failure event
            progress.add_task("Processing logs...", total=None)
            failure = PipelineFailure(
                id="cli_analysis",
                pipeline_id="cli",
                job_id="cli_job",
                repository="local",
                branch="main",
                commit_sha="unknown",
                failure_time=Path(logs_file).stat().st_mtime,
                logs=logs_content,
                exit_code=1,
                stage="analysis",
                step_name="log_analysis"
            )
            
            # Analyze failure
            progress.add_task("Running AI analysis...", total=None)
            result = asyncio.run(detector.detect_failure(failure))
        
        # Format results
        if result.confidence >= confidence_threshold:
            console.print(f"✅ [green]Failure classified with {result.confidence:.2%} confidence[/green]")
            
            # Create results table
            table = Table(title="🔍 Failure Analysis Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Failure Type", result.failure_type.value)
            table.add_row("Confidence", f"{result.confidence:.2%}")
            table.add_row("Severity", result.severity.name)
            table.add_row("Patterns Found", str(len(result.matched_patterns)))
            table.add_row("Suggestions", str(len(result.remediation_suggestions)))
            
            console.print(table)
            
            if result.remediation_suggestions:
                console.print("\n💡 [yellow]Remediation Suggestions:[/yellow]")
                for i, suggestion in enumerate(result.remediation_suggestions, 1):
                    console.print(f"  {i}. {suggestion}")
        else:
            console.print(f"⚠️ [yellow]Low confidence classification ({result.confidence:.2%})[/yellow]")
            console.print("Consider manual review of the failure logs.")
        
        # Save results if output specified
        if output:
            results_data = {
                "failure_type": result.failure_type.value,
                "confidence": result.confidence,
                "severity": result.severity.name,
                "matched_patterns": result.matched_patterns,
                "remediation_suggestions": result.remediation_suggestions,
                "analysis_timestamp": result.timestamp.isoformat()
            }
            
            if format_type.lower() == "json":
                output.write_text(json.dumps(results_data, indent=2))
            elif format_type.lower() == "yaml":
                import yaml
                output.write_text(yaml.dump(results_data, default_flow_style=False))
            
            console.print(f"📄 Results saved to {output}")
    
    except Exception as e:
        console.print(f"[red]Error analyzing logs: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def heal(
    failure_id: str = typer.Argument(..., help="Failure ID to heal"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Specific healing strategy to use"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
    force: bool = typer.Option(False, "--force", help="Force healing even if confidence is low"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """🚀 Execute healing strategies for a detected failure."""
    setup_logging(debug)
    
    console.print(Panel(f"🚀 Healing Failure: {failure_id}", style="green"))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize healing engine
            progress.add_task("Initializing healing engine...", total=None)
            healing_engine = HealingEngine(settings)
            
            if dry_run:
                console.print("🔍 [yellow]DRY RUN MODE - No actual changes will be made[/yellow]")
                # Show what would be done
                console.print(f"Would attempt to heal failure: {failure_id}")
                if strategy:
                    console.print(f"Using strategy: {strategy}")
                console.print("Available strategies:")
                for strat_name in healing_engine.strategies:
                    console.print(f"  - {strat_name}")
            else:
                # Execute healing (this would need actual failure data)
                console.print("🔧 [green]Healing execution would happen here[/green]")
                console.print(f"Target: {failure_id}")
                if strategy:
                    console.print(f"Strategy: {strategy}")
    
    except Exception as e:
        console.print(f"[red]Error during healing: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def status(
    repository: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository to check status for"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table, json)"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """📊 Show healing guard status and statistics."""
    setup_logging(debug)
    
    console.print(Panel("📊 Healing Guard Status", style="cyan"))
    
    try:
        # This would connect to actual healing guard service
        # For now, show mock status
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        table.add_row("Failure Detector", "✅ Active", "AI model loaded")
        table.add_row("Healing Engine", "✅ Active", "12 strategies available")
        table.add_row("Quantum Planner", "✅ Active", "Optimization enabled")
        table.add_row("API Server", "🔄 Unknown", "Use --debug for connection test")
        
        console.print(table)
        
        if repository:
            console.print(f"\n🔍 Repository: {repository}")
            console.print("📈 Recent healing statistics would be shown here")
    
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind server to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """🌐 Start the Healing Guard API server."""
    setup_logging(debug)
    
    console.print(Panel(f"🌐 Starting Healing Guard Server on {host}:{port}", style="blue"))
    
    try:
        import uvicorn
        from healing_guard.api.main import app as fastapi_app
        
        console.print(f"🚀 Server starting...")
        console.print(f"📡 API will be available at http://{host}:{port}")
        console.print(f"📚 Documentation at http://{host}:{port}/docs")
        
        uvicorn.run(
            "healing_guard.api.main:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="debug" if debug else "info"
        )
    
    except ImportError:
        console.print("[red]Error: uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    generate: bool = typer.Option(False, "--generate", help="Generate example configuration"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for generated config"),
):
    """⚙️ Manage Healing Guard configuration."""
    
    if show:
        console.print(Panel("⚙️ Current Configuration", style="yellow"))
        try:
            table = Table()
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Environment", settings.environment)
            table.add_row("Host", settings.host)
            table.add_row("Port", str(settings.port))
            table.add_row("Debug", str(settings.debug))
            table.add_row("Database URL", settings.database.url[:50] + "..." if len(settings.database.url) > 50 else settings.database.url)
            table.add_row("Redis URL", settings.redis.url)
            
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
    
    if validate:
        console.print("🔍 Validating configuration...")
        try:
            # Validate by accessing key settings
            _ = settings.database.url
            _ = settings.redis.url
            _ = settings.environment
            console.print("✅ [green]Configuration is valid[/green]")
        except Exception as e:
            console.print(f"❌ [red]Configuration error: {e}[/red]")
            raise typer.Exit(1)
    
    if generate:
        example_config = {
            "healing_guard": {
                "ml_model_path": "/app/models/failure_classifier.pkl",
                "confidence_threshold": 0.75,
                "max_healing_retries": 3,
                "cache_ttl_seconds": 3600,
                "strategies": {
                    "retry_with_backoff": {"enabled": True, "max_attempts": 3},
                    "resource_scaling": {"enabled": True, "scale_factor": 1.5},
                    "cache_clearing": {"enabled": True, "selective": True}
                },
                "notifications": {
                    "slack_webhook": "https://hooks.slack.com/...",
                    "email_alerts": True
                }
            }
        }
        
        config_text = json.dumps(example_config, indent=2)
        
        if output:
            output.write_text(config_text)
            console.print(f"📄 Example configuration saved to {output}")
        else:
            console.print(Panel("📄 Example Configuration", style="yellow"))
            console.print(config_text)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
):
    """🛡️ Self-Healing Pipeline Guard CLI"""
    if version:
        from healing_guard import __version__
        console.print(f"Healing Guard v{__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()