# Install the bixverse.gpu agent skill

bixverse.gpu ships a skill that teaches coding agents how to *use* the
package: which GPU function replaces which bixverse step, the knobs that
differ, what does not run on the GPU and the traps. This copies it out
of the installed package to where your agent looks for it. It
complements the bixverse skill, see
[`bixverse::install_agent_skill()`](https://gregorlueg.github.io/bixverse/reference/install_agent_skill.html).

The skill is versioned with the package, so re-run this after upgrading
bixverse.gpu to keep the agent in sync with the code.

## Usage

``` r
install_agent_skill_gpu(
  dest = NULL,
  agent = c("claude", "codex", "generic"),
  overwrite = FALSE
)
```

## Arguments

- dest:

  String or `NULL`. Directory to install into. `NULL` picks the default
  for `agent`: `"~/.claude/skills"` for `"claude"` and
  `"~/.codex/skills"` for `"codex"`. Required for `"generic"`. The skill
  lands in a `bixverse-gpu` subdirectory of this.

- agent:

  String. One of `c("claude", "codex", "generic")`. Controls the name of
  the entry file and whether the frontmatter is kept.

- overwrite:

  Boolean. Replace an existing installation. Defaults to `FALSE`, in
  which case an existing `bixverse-gpu` directory causes an error rather
  than a silent clobber.

## Value

The path the skill was written to, invisibly.

## Details

The reference files are plain markdown and work anywhere. Only the entry
file differs between agents:

- `"claude"` writes `SKILL.md` with the YAML frontmatter Claude Code
  uses to decide when to load the skill. Discovery is automatic.

- `"codex"` and `"generic"` write `AGENTS.md` with the frontmatter
  stripped, since no other agent reads it. There is no auto-discovery,
  so point the agent at the directory yourself.

## Examples

``` r
if (FALSE) { # \dontrun{
# Claude Code, into ~/.claude/skills, auto-discovered
install_agent_skill_gpu()

# Codex, into ~/.codex/skills as AGENTS.md
install_agent_skill_gpu(agent = "codex")

# refresh after a package upgrade
install_agent_skill_gpu(overwrite = TRUE)
} # }
```
