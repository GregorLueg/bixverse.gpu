# agent skill ------------------------------------------------------------------

#' Install the bixverse.gpu agent skill
#'
#' @description
#' bixverse.gpu ships a skill that teaches coding agents how to *use* the
#' package: which GPU function replaces which bixverse step, the knobs that
#' differ, what does not run on the GPU and the traps. This copies it out of
#' the installed package to where your agent looks for it. It complements the
#' bixverse skill, see [bixverse::install_agent_skill()].
#'
#' The skill is versioned with the package, so re-run this after upgrading
#' bixverse.gpu to keep the agent in sync with the code.
#'
#' @details
#' The reference files are plain markdown and work anywhere. Only the entry file
#' differs between agents:
#' \itemize{
#'   \item `"claude"` writes `SKILL.md` with the YAML frontmatter Claude Code
#'   uses to decide when to load the skill. Discovery is automatic.
#'   \item `"codex"` and `"generic"` write `AGENTS.md` with the frontmatter
#'   stripped, since no other agent reads it. There is no auto-discovery, so
#'   point the agent at the directory yourself.
#' }
#'
#' @param dest String or `NULL`. Directory to install into. `NULL` picks the
#' default for `agent`: `"~/.claude/skills"` for `"claude"` and
#' `"~/.codex/skills"` for `"codex"`. Required for `"generic"`. The skill lands
#' in a `bixverse-gpu` subdirectory of this.
#' @param agent String. One of `c("claude", "codex", "generic")`. Controls the
#' name of the entry file and whether the frontmatter is kept.
#' @param overwrite Boolean. Replace an existing installation. Defaults to
#' `FALSE`, in which case an existing `bixverse-gpu` directory causes an error
#' rather than a silent clobber.
#'
#' @returns The path the skill was written to, invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Claude Code, into ~/.claude/skills, auto-discovered
#' install_agent_skill_gpu()
#'
#' # Codex, into ~/.codex/skills as AGENTS.md
#' install_agent_skill_gpu(agent = "codex")
#'
#' # refresh after a package upgrade
#' install_agent_skill_gpu(overwrite = TRUE)
#' }
install_agent_skill_gpu <- function(
  dest = NULL,
  agent = c("claude", "codex", "generic"),
  overwrite = FALSE
) {
  agent <- match.arg(agent)

  # checks
  checkmate::qassert(dest, c("S1", "0"))
  checkmate::qassert(overwrite, "B1")

  if (is.null(dest)) {
    dest <- switch(
      agent,
      claude = "~/.claude/skills",
      codex = "~/.codex/skills",
      generic = stop("`dest` is required when `agent = \"generic\"`.")
    )
  }

  source_dir <- system.file("skills", "bixverse-gpu", package = "bixverse.gpu")
  if (source_dir == "") {
    stop(paste(
      "The agent skill was not found in the installed package. This usually",
      "means bixverse.gpu was installed from a source tree predating the skill."
    ))
  }

  dest <- path.expand(dest)
  target <- file.path(dest, "bixverse-gpu")

  if (dir.exists(target)) {
    if (!overwrite) {
      stop(sprintf(
        "A bixverse-gpu skill already exists at %s. Re-run with `overwrite = TRUE`.",
        target
      ))
    }
    unlink(target, recursive = TRUE, force = TRUE)
  }

  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  # copies source_dir itself (not just its contents), giving dest/bixverse-gpu
  ok <- file.copy(source_dir, dest, recursive = TRUE)
  if (!ok || !file.exists(file.path(target, "SKILL.md"))) {
    stop(sprintf("Failed to install the agent skill to %s", target))
  }

  entry <- file.path(target, "SKILL.md")

  if (agent != "claude") {
    lines <- bixverse:::.strip_frontmatter(readLines(entry, warn = FALSE))
    writeLines(lines, file.path(target, "AGENTS.md"))
    unlink(entry)
    entry <- file.path(target, "AGENTS.md")
  }

  message(sprintf("bixverse.gpu agent skill installed to %s", target))
  message(switch(
    agent,
    claude = "Start a new Claude Code session to pick it up.",
    sprintf(
      paste(
        "No auto-discovery outside Claude Code. Point your agent at %s,",
        "or symlink it into the project you are working in."
      ),
      entry
    )
  ))

  invisible(target)
}
