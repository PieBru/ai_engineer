* **`/rules <subcommand> [arguments]`** - Manage the AI's guiding rules (system prompt components).
  Rules are markdown files. Active rules reside in the `./.aie_rules/` directory and are compiled into the system prompt.
  All available rules (both active and inactive) are stored in the `./.aie_rules_all/` directory.
  Rules are applied in lexicographical order of their filenames (e.g., `001_rule.md` before `002_rule.md`).

Subcommands:
  * **`show`**: Displays the complete, ordered content of all currently active rules. This shows the exact rule-based instructions being used by the AI.
  * **`list [all|enabled|disabled]`**:
      * `list` or `list enabled` (default): Lists active rule files from `./.aie_rules/`, along with their descriptions.
      * `list disabled`: Lists rule files in `./.aie_rules_all/` that are not currently active, with their descriptions.
      * `list all`: Lists all rule files from `./.aie_rules_all/`, indicating their status (e.g., `[active]`, `[available]`) and descriptions.
      *(Rule descriptions are read from YAML frontmatter within each .md rule file, e.g., a `description:` field.)*
  * **`enable <rule_pattern>`**: Activates rule(s) by copying them from `./.aie_rules_all/` to `./.aie_rules/`. Supports wildcards (`*`, `?`) in `<rule_pattern>`.
      Example: `/rules enable python_*.md`
  * **`disable <rule_pattern>`**: Deactivates rule(s) by deleting them from `./.aie_rules/`. The master copy remains in `./.aie_rules_all/`. Supports wildcards.
      Example: `/rules disable 005_*`
  * **`reset`**: Resets the active rule set to system defaults. This typically involves:
      1. Clearing all files from `./.aie_rules/`.
      2. Copying predefined system rules (e.g., `000_rules_header.md`, `999_rules_trailer.md`) from `./.aie_rules_all/` to `./.aie_rules/`.
