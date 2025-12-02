//! Terminal output handling
//!
//! Handles writing frames to the terminal using crossterm.

use catsith_core::TerminalFrame;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{self, Write};
use thiserror::Error;

/// Terminal output errors
#[derive(Debug, Error)]
pub enum OutputError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Terminal not available")]
    NotAvailable,
}

/// Terminal output handler
pub struct TerminalOutput {
    /// Whether we're in alternate screen mode
    alternate_screen: bool,
    /// Whether cursor is hidden
    cursor_hidden: bool,
    /// Last frame for diff rendering
    last_frame: Option<TerminalFrame>,
    /// Enable diff rendering (only update changed cells)
    diff_mode: bool,
}

impl TerminalOutput {
    /// Create a new terminal output handler
    pub fn new() -> Self {
        Self {
            alternate_screen: false,
            cursor_hidden: false,
            last_frame: None,
            diff_mode: true,
        }
    }

    /// Enable or disable diff mode
    pub fn set_diff_mode(&mut self, enabled: bool) {
        self.diff_mode = enabled;
        if !enabled {
            self.last_frame = None;
        }
    }

    /// Initialize the terminal for rendering
    pub fn init(&mut self) -> Result<(), OutputError> {
        let mut stdout = io::stdout();

        // Enter alternate screen
        execute!(stdout, EnterAlternateScreen)?;
        self.alternate_screen = true;

        // Hide cursor
        execute!(stdout, Hide)?;
        self.cursor_hidden = true;

        // Clear screen
        execute!(stdout, Clear(ClearType::All))?;

        // Enable raw mode
        terminal::enable_raw_mode()?;

        Ok(())
    }

    /// Cleanup terminal state
    pub fn cleanup(&mut self) -> Result<(), OutputError> {
        let mut stdout = io::stdout();

        // Disable raw mode
        let _ = terminal::disable_raw_mode();

        // Show cursor
        if self.cursor_hidden {
            execute!(stdout, Show)?;
            self.cursor_hidden = false;
        }

        // Leave alternate screen
        if self.alternate_screen {
            execute!(stdout, LeaveAlternateScreen)?;
            self.alternate_screen = false;
        }

        Ok(())
    }

    /// Render a frame to the terminal
    pub fn render(&mut self, frame: &TerminalFrame) -> Result<(), OutputError> {
        let mut stdout = io::stdout();

        if self.diff_mode && self.last_frame.is_some() {
            self.render_diff(&mut stdout, frame)?;
        } else {
            self.render_full(&mut stdout, frame)?;
        }

        stdout.flush()?;

        // Store for diff
        if self.diff_mode {
            self.last_frame = Some(frame.clone());
        }

        Ok(())
    }

    /// Render entire frame
    fn render_full<W: Write>(&self, out: &mut W, frame: &TerminalFrame) -> Result<(), OutputError> {
        execute!(out, MoveTo(0, 0))?;

        let mut last_fg: Option<[u8; 3]> = None;
        let mut last_bg: Option<[u8; 3]> = None;

        for y in 0..frame.height {
            execute!(out, MoveTo(0, y as u16))?;

            for x in 0..frame.width {
                if let Some(cell) = frame.get(x, y) {
                    // Update foreground color if changed
                    if last_fg != Some(cell.fg) {
                        execute!(
                            out,
                            SetForegroundColor(Color::Rgb {
                                r: cell.fg[0],
                                g: cell.fg[1],
                                b: cell.fg[2],
                            })
                        )?;
                        last_fg = Some(cell.fg);
                    }

                    // Update background color if changed
                    if last_bg != cell.bg {
                        if let Some(bg) = cell.bg {
                            execute!(
                                out,
                                SetBackgroundColor(Color::Rgb {
                                    r: bg[0],
                                    g: bg[1],
                                    b: bg[2],
                                })
                            )?;
                        } else {
                            execute!(out, ResetColor)?;
                            last_fg = None; // Reset clears both
                        }
                        last_bg = cell.bg;
                    }

                    execute!(out, Print(cell.char))?;
                }
            }
        }

        execute!(out, ResetColor)?;
        Ok(())
    }

    /// Render only changed cells
    fn render_diff<W: Write>(&self, out: &mut W, frame: &TerminalFrame) -> Result<(), OutputError> {
        let last = self.last_frame.as_ref().unwrap();

        for y in 0..frame.height {
            for x in 0..frame.width {
                let current = frame.get(x, y);
                let previous = last.get(x, y);

                // Only render if cell changed
                if current != previous {
                    if let Some(cell) = current {
                        execute!(out, MoveTo(x as u16, y as u16))?;

                        execute!(
                            out,
                            SetForegroundColor(Color::Rgb {
                                r: cell.fg[0],
                                g: cell.fg[1],
                                b: cell.fg[2],
                            })
                        )?;

                        if let Some(bg) = cell.bg {
                            execute!(
                                out,
                                SetBackgroundColor(Color::Rgb {
                                    r: bg[0],
                                    g: bg[1],
                                    b: bg[2],
                                })
                            )?;
                        }

                        execute!(out, Print(cell.char))?;
                    }
                }
            }
        }

        execute!(out, ResetColor)?;
        Ok(())
    }

    /// Get terminal size
    pub fn size() -> Result<(u32, u32), OutputError> {
        let (cols, rows) = terminal::size()?;
        Ok((cols as u32, rows as u32))
    }

    /// Check if terminal supports true color
    pub fn supports_true_color() -> bool {
        // Check COLORTERM environment variable
        if let Ok(colorterm) = std::env::var("COLORTERM") {
            return colorterm == "truecolor" || colorterm == "24bit";
        }

        // Check TERM
        if let Ok(term) = std::env::var("TERM") {
            return term.contains("256color") || term.contains("truecolor");
        }

        false
    }
}

impl Default for TerminalOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TerminalOutput {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Simple function to print a frame to stdout (no alternate screen)
pub fn print_frame(frame: &TerminalFrame) {
    print!("{}", frame.to_ansi());
}

/// Print frame to a string
pub fn frame_to_string(frame: &TerminalFrame) -> String {
    frame.to_ansi()
}

#[cfg(test)]
mod tests {
    use super::*;
    use catsith_core::TerminalCell;

    #[test]
    fn test_frame_to_string() {
        let mut frame = TerminalFrame::new(10, 3);
        frame.set(5, 1, TerminalCell::new('X').with_fg([255, 0, 0]));

        let output = frame_to_string(&frame);
        assert!(output.contains('X'));
        assert!(output.contains("\x1b[")); // Contains ANSI escape
    }

    #[test]
    fn test_terminal_size() {
        // This might fail in CI without a terminal
        let _ = TerminalOutput::size();
    }

    #[test]
    fn test_diff_mode() {
        let mut output = TerminalOutput::new();
        assert!(output.diff_mode);

        output.set_diff_mode(false);
        assert!(!output.diff_mode);
        assert!(output.last_frame.is_none());
    }
}
