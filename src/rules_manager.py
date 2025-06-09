# src/rules_manager.py
import os
import shutil
from pathlib import Path
import glob
import yaml # PyYAML: Add to requirements.txt
from rich.panel import Panel
from rich.text import Text

from src.app_state import AppState # Assuming AppState is directly usable
from src.prompts import RichMarkdown # For displaying rule content

# Define rule directories (relative to project root)
# These could be made configurable later if needed.
RULES_DIR_ACTIVE = Path("./.aie_rules")
RULES_DIR_ALL = Path("./.aie_rules_all")
SYSTEM_RULES_FILENAMES = ["000_rules_header.md", "999_rules_trailer.md"]

# Configuration key for rule application order (if you implement prepend/append to user message)
# For now, we assume rules form the system prompt.
# RULE_APPLICATION_ORDER_CONFIG_KEY = "RULE_APPLICATION_ORDER"
# DEFAULT_RULE_APPLICATION_ORDER = "prepend"

def ensure_rule_directories_exist(app_state: AppState):
    """Creates rule directories if they don't exist."""
    RULES_DIR_ACTIVE.mkdir(parents=True, exist_ok=True)
    RULES_DIR_ALL.mkdir(parents=True, exist_ok=True)
    if app_state.DEBUG_RULES:
        app_state.console.print(f"[dim]Ensured rule directories exist: {RULES_DIR_ACTIVE}, {RULES_DIR_ALL}[/dim]")

def _get_rule_description(rule_file_path: Path) -> str:
    """Parses YAML frontmatter to get the rule description."""
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.startswith("---"):
            end_frontmatter = content.find("---", 3)
            if end_frontmatter != -1:
                frontmatter_str = content[3:end_frontmatter]
                try:
                    metadata = yaml.safe_load(frontmatter_str)
                    return metadata.get("description", "No description provided.")
                except yaml.YAMLError as e:
                    return f"Error parsing description: {e}"
        return "No description found in frontmatter."
    except Exception as e:
        return f"Error reading rule file: {e}"

def _get_sorted_active_rule_paths(app_state: AppState) -> list[Path]:
    """Gets a sorted list of active rule file paths."""
    ensure_rule_directories_exist(app_state)
    return sorted(RULES_DIR_ACTIVE.glob("*.md*")) # Ensure glob pattern matches .md and .md* if needed

def _get_rule_content_without_frontmatter(rule_file_path: Path, app_state: AppState) -> str:
    """Reads rule content, stripping YAML frontmatter."""
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.startswith("---"):
                end_frontmatter = content.find("---", 3)
                if end_frontmatter != -1:
                    # Content after the second '---' and any leading newlines
                    return content[end_frontmatter + 3:].lstrip()
            return content # No frontmatter found or malformed
    except Exception as e:
        if app_state and app_state.console:
            app_state.console.print(f"[red]Error reading content of rule file {rule_file_path.name}: {e}[/red]")
        return ""

def update_effective_system_prompt(app_state: AppState):
    """
    Reads all active rules, concatenates them, and updates app_state.system_prompt.
    This replaces the static loading of _default.md.
    """
    ensure_rule_directories_exist(app_state)
    active_rule_files = _get_sorted_active_rule_paths(app_state)
    
    concatenated_rules_content = []
    for rule_file in active_rule_files:
        concatenated_rules_content.append(_get_rule_content_without_frontmatter(rule_file, app_state))
    
    # Join with double newline for better separation if rules are multi-paragraph
    app_state.system_prompt = "\n\n".join(filter(None, concatenated_rules_content)) 
    
    if app_state.DEBUG_RULES:
         app_state.console.print(f"[dim]Updated effective system prompt from {len(active_rule_files)} active rules. Length: {len(app_state.system_prompt)} chars.[/dim]")

def initialize_rules_system(app_state: AppState):
    """
    Ensures rule directories exist and system rules are active.
    This should be called on application startup.
    """
    ensure_rule_directories_exist(app_state)
    
    # Ensure system rules are present in .aie_rules/
    # These are copied from .aie_rules_all/
    for rule_filename in SYSTEM_RULES_FILENAMES:
        source_path = RULES_DIR_ALL / rule_filename
        dest_path = RULES_DIR_ACTIVE / rule_filename
        
        if source_path.exists():
            # Copy (and overwrite) to ensure active system rules match the master versions
            try:
                shutil.copy2(source_path, dest_path)
                if app_state.DEBUG_RULES:
                    app_state.console.print(f"[dim]System rule '{rule_filename}' ensured in active rules.[/dim]")
            except Exception as e:
                app_state.console.print(f"[red]Error copying system rule {rule_filename}: {e}[/red]")
        else:
            # This is a problem, system rules should exist in _all
            app_state.console.print(f"[yellow]Warning: System rule '{rule_filename}' not found in {RULES_DIR_ALL}. It might be missing or misnamed.[/yellow]")
            # Consider creating an empty placeholder in .aie_rules if critical, or logging an error.
            # For now, if it's missing in _all, it won't be in _active either.

    update_effective_system_prompt(app_state) # Build the initial system prompt

# --- Command Implementations called by command_handlers.py ---

def show_active_rules_command(app_state: AppState):
    """Handles '/rules show' command."""
    update_effective_system_prompt(app_state) # Ensure it's fresh
    
    if not app_state.system_prompt.strip():
        app_state.console.print("[yellow]No active rules found or all active rules are empty.[/yellow]")
        return

    # Using RichMarkdown for potentially complex markdown in rules
    app_state.console.print(Panel(
        RichMarkdown(app_state.system_prompt),
        title="[bold blue]📝 Active System Rules[/bold blue]",
        border_style="blue",
        padding=(1, 1)
    ))

def list_rules_command(app_state: AppState, list_filter: str):
    """Handles '/rules list [enabled|disabled|all]' command."""
    ensure_rule_directories_exist(app_state)
    list_filter = list_filter.lower()
    
    output_text = Text()

    active_rule_paths = _get_sorted_active_rule_paths(app_state)
    active_rule_names = {p.name for p in active_rule_paths}
    all_master_rule_paths = sorted(RULES_DIR_ALL.glob("*.md*"))

    if list_filter == "enabled" or list_filter == "all":
        output_text.append("Active Rules (in ./.aie_rules/):\n", style="bold green")
        if not active_rule_paths:
            output_text.append("  No active rules.\n")
        for rule_file_path in active_rule_paths:
            desc = _get_rule_description(rule_file_path)
            output_text.append(f"  - {rule_file_path.name}", style="white")
            output_text.append(f" : {desc}\n", style="dim")
    
    if list_filter == "disabled" or list_filter == "all":
        if list_filter == "all": # Add a separator if both sections are printed
             output_text.append("\n")
        output_text.append("Available Rules (in ./.aie_rules_all/):\n", style="bold yellow")
        
        found_any_in_all = False
        displayed_disabled_count = 0
        for rule_file_path in all_master_rule_paths:
            found_any_in_all = True
            is_active = rule_file_path.name in active_rule_names
            desc = _get_rule_description(rule_file_path)
            
            if list_filter == "all":
                status_style = "green" if is_active else "dim"
                status_text = "[active]" if is_active else "[available]"
                output_text.append(f"  - {rule_file_path.name} ", style="white")
                output_text.append(status_text, style=status_style)
                output_text.append(f" : {desc}\n", style="dim")
            elif not is_active: # list_filter == "disabled"
                output_text.append(f"  - {rule_file_path.name}", style="white")
                output_text.append(f" : {desc}\n", style="dim")
                displayed_disabled_count +=1
        
        if not found_any_in_all:
            output_text.append("  No rules found in .aie_rules_all/ directory.\n")
        elif list_filter == "disabled" and displayed_disabled_count == 0:
             output_text.append("  All available rules are currently active.\n")

    app_state.console.print(output_text)

def enable_rules_command(app_state: AppState, rule_pattern: str):
    """Handles '/rules enable <rule_pattern>' command."""
    ensure_rule_directories_exist(app_state)
    
    matched_source_files = list(RULES_DIR_ALL.glob(rule_pattern))
    if not matched_source_files:
        app_state.console.print(f"[yellow]No rules matching pattern '{rule_pattern}' found in {RULES_DIR_ALL}.[/yellow]")
        return

    enabled_count = 0
    for src_file_path in matched_source_files:
        if src_file_path.is_file(): # Ensure it's a file, not a dir matching pattern
            dest_file_path = RULES_DIR_ACTIVE / src_file_path.name
            try:
                shutil.copy2(src_file_path, dest_file_path) # Overwrites if exists
                app_state.console.print(f"[green]✓ Enabled rule: {src_file_path.name}[/green]")
                enabled_count += 1
            except Exception as e:
                app_state.console.print(f"[red]Error enabling rule {src_file_path.name}: {e}[/red]")
    
    if enabled_count > 0:
        update_effective_system_prompt(app_state)
    elif matched_source_files: # Files matched but were not files (e.g. dirs) or other issue
        app_state.console.print(f"[yellow]Pattern '{rule_pattern}' matched items, but no rule files were enabled.[/yellow]")


def disable_rules_command(app_state: AppState, rule_pattern: str):
    """Handles '/rules disable <rule_pattern>' command."""
    ensure_rule_directories_exist(app_state)
    
    matched_active_files = list(RULES_DIR_ACTIVE.glob(rule_pattern))
    if not matched_active_files:
        app_state.console.print(f"[yellow]No active rules matching pattern '{rule_pattern}' found to disable.[/yellow]")
        return

    disabled_count = 0
    for file_path in matched_active_files:
        if file_path.is_file():
            # System rules can be disabled like any other rule. `reset` will bring them back.
            try:
                file_path.unlink()
                app_state.console.print(f"[green]✓ Disabled rule: {file_path.name}[/green]")
                disabled_count += 1
            except Exception as e:
                app_state.console.print(f"[red]Error disabling rule {file_path.name}: {e}[/red]")
                
    if disabled_count > 0:
        update_effective_system_prompt(app_state)
    elif matched_active_files:
        app_state.console.print(f"[yellow]Pattern '{rule_pattern}' matched items, but no rule files were disabled.[/yellow]")

def reset_rules_command(app_state: AppState):
    """Handles '/rules reset' command."""
    ensure_rule_directories_exist(app_state)
    
    # Consider adding a confirmation prompt here if desired
    # response = app_state.prompt_session.prompt("Are you sure you want to reset all active rules to system defaults? (yes/no): ")
    # if response.lower() != 'yes':
    #     app_state.console.print("[yellow]Rule reset cancelled.[/yellow]")
    #     return

    app_state.console.print("[yellow]Resetting active rules to system defaults...[/yellow]")
    
    # 1. Clear all files from ./.aie_rules/
    for item in RULES_DIR_ACTIVE.iterdir():
        if item.is_file(): # Only delete files
            try:
                item.unlink()
            except Exception as e:
                app_state.console.print(f"[red]Error deleting rule file {item.name} during reset: {e}[/red]")
        
    if app_state.DEBUG_RULES:
        app_state.console.print(f"[dim]Cleared active rules directory: {RULES_DIR_ACTIVE}[/dim]")
    
    # 2. Re-initialize system rules and update prompt (initialize_rules_system does both)
    initialize_rules_system(app_state) 
    
    app_state.console.print("[green]✓ Rules reset to system defaults.[/green]")
