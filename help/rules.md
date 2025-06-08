* **`/rules <subcommand> [arguments]`** - Manage the AI's guiding rules (system prompt). Call /rules without arguments for detailed usage.
Subcommands:
   * **`show`**: Display the current system prompt (rules) being used.
   * **`list`**: List available rule files in the ./.aie_rules/ directory.                                           │
   * **`add <rule-file>`**: Add rules from a specified markdown file to the current session's system prompt.
   * **`reset`**: Empties the current system prompt, then asks for confirmation to load default rules from `./.aie_rules/_default.md`.
