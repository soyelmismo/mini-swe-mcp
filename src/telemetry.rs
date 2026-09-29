//! Telemetry setup for the binary: the terse stderr log format and the
//! non-blocking sink that keeps STDOUT free for the MCP JSON-RPC stream.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

struct ShortFormatter;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for ShortFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let meta = event.metadata();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let secs = now % 60;
        let mins = (now / 60) % 60;
        let hours = (now / 3600) % 24;

        let target = meta.target();
        let short_target = target.strip_prefix("mini_swe_mcp::").unwrap_or(target);

        let lvl = match *meta.level() {
            tracing::Level::ERROR => "\x1b[31mERRO\x1b[0m",
            tracing::Level::WARN => "\x1b[33mWARN\x1b[0m",
            tracing::Level::INFO => "\x1b[32mINFO\x1b[0m",
            tracing::Level::DEBUG => "\x1b[34mDEBG\x1b[0m",
            tracing::Level::TRACE => "\x1b[35mTRCE\x1b[0m",
        };

        write!(writer, "{:02}:{:02}:{:02} {} [{}] ", hours, mins, secs, lvl, short_target)?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Non-blocking stderr writer for tracing.
struct AsyncStderrWriter {
    tx: std::sync::mpsc::SyncSender<String>,
}

impl std::io::Write for AsyncStderrWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        self.tx
            .send(text.into_owned())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "log sink gone"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for AsyncStderrWriter {
    type Writer = AsyncStderrWriter;

    fn make_writer(&'a self) -> Self::Writer {
        AsyncStderrWriter {
            tx: self.tx.clone(),
        }
    }
}

fn non_blocking_stderr() -> AsyncStderrWriter {
    // Bounded so a burst of telemetry can never stall a runtime thread; lines
    // are dropped when the sink is full, which is fine for diagnostics.
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(1024);
    std::thread::Builder::new()
        .name("telemetry-stderr".to_string())
        .spawn(move || {
            use std::io::Write as _;
            let mut stderr = std::io::stderr();
            while let Ok(line) = rx.recv() {
                if stderr.write_all(line.as_bytes()).is_err() {
                    break;
                }
                let _ = stderr.flush();
            }
        })
        .ok();
    AsyncStderrWriter { tx }
}

/// Install the stderr telemetry sink.
///
/// STDOUT is dedicated to MCP JSON-RPC, so every log line goes to STDERR, and
/// the sink is non-blocking so telemetry can never stall a runtime thread. In
/// stdio mode the default level drops to `WARN` to keep the channel quiet.
pub fn init(stdio_mode: bool) {
    let default_level = if stdio_mode {
        tracing::Level::WARN
    } else {
        tracing::Level::INFO
    };

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::default().add_directive(default_level.into()));

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .event_format(ShortFormatter)
                .with_writer(non_blocking_stderr()),
        )
        .with(env_filter)
        .init();
}
