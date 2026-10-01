from pathlib import Path
p = Path("src/pool/runner/turn.rs")
s = p.read_text()

def sub(s, old, new):
    assert old in s, old[:80]
    return s.replace(old, new, 1)

s = sub(s, """    /// Fold one turn's repository sample in, remember the command the turn
    /// ran, and report the nudge, if any, the streak has earned. Each of the
    /// three thresholds fires once per streak and in order, so a streak of
    /// unchanged worktree reaches the plan and then the pause no matter where
    /// its thresholds sit.
    ///
    /// `None` is a sample git could not answer: it is not evidence of progress,
    /// so it neither extends nor resets the streak.
    fn record(
        &mut self,
        sample: Option<String>,
        limits: ReadOnlyThresholds,
        command: &str,
    ) -> Option<ReadOnlyNudge> {
        self.note_command(command);
        let sample = sample?;""",
"""    /// Fold one turn's repository sample in and report the nudge, if any, the
    /// streak has earned. Each of the three thresholds fires once per streak and
    /// in order, so a streak of unchanged worktree reaches the plan and then the
    /// pause no matter where its thresholds sit.
    ///
    /// `None` is a sample git could not answer: it is not evidence of progress,
    /// so it neither extends nor resets the streak.
    fn record(
        &mut self,
        sample: Option<String>,
        limits: ReadOnlyThresholds,
    ) -> Option<ReadOnlyNudge> {
        let sample = sample?;""")

s = sub(s, """    /// Folded in by [`ReadOnlyStreak::record`] rather than at execution time:
    /// the detector runs before this turn's command is registered, so a pause
    /// it raises would otherwise report that nothing had been read.""",
"""    /// Folded in by [`ProgressWatch::register_command`], the same place the
    /// repetition detector sees each command, rather than at execution time.""")

s = sub(s, """        // The detector runs before this turn's command is registered, so the
        // command it is told about is the one the streak last read: the pause
        // it may raise has to report what the worker spent its turns on.
        let last_command = self
            .watch
            .last_command
            .clone()
            .unwrap_or_default();
        let Some(nudge) = self
            .watch
            .read_only
            .record(sample, limits, &last_command)
        else {
            return Ok(());
        };""",
"""        let Some(nudge) = self.watch.read_only.record(sample, limits) else {
            return Ok(());
        };""")

p.write_text(s)
print("ok")
