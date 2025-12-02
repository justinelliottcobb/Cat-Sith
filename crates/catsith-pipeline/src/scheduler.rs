//! Render scheduling
//!
//! Manages frame timing, quality adjustment, and render prioritization.

use catsith_core::QualityTier;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Target frame rate
    pub target_fps: f64,
    /// Minimum acceptable frame rate
    pub min_fps: f64,
    /// Enable adaptive quality
    pub adaptive_quality: bool,
    /// Frame time history size for averaging
    pub history_size: usize,
    /// Quality adjustment threshold (how much over budget triggers adjustment)
    pub adjustment_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            min_fps: 30.0,
            adaptive_quality: true,
            history_size: 30,
            adjustment_threshold: 1.2, // 20% over budget
        }
    }
}

/// Render scheduler
pub struct RenderScheduler {
    config: SchedulerConfig,
    /// Frame time history (in milliseconds)
    frame_times: VecDeque<f64>,
    /// Current quality tier
    current_quality: QualityTier,
    /// Last frame start time
    last_frame_start: Option<Instant>,
    /// Frames rendered
    frames_rendered: u64,
    /// Frames dropped
    frames_dropped: u64,
}

impl RenderScheduler {
    /// Create a new scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            frame_times: VecDeque::with_capacity(30),
            current_quality: QualityTier::Medium,
            last_frame_start: None,
            frames_rendered: 0,
            frames_dropped: 0,
        }
    }

    /// Get target frame time in milliseconds
    pub fn target_frame_time_ms(&self) -> f64 {
        1000.0 / self.config.target_fps
    }

    /// Get minimum acceptable frame time in milliseconds
    pub fn min_frame_time_ms(&self) -> f64 {
        1000.0 / self.config.min_fps
    }

    /// Start a new frame
    pub fn begin_frame(&mut self) {
        self.last_frame_start = Some(Instant::now());
    }

    /// End the current frame and record timing
    pub fn end_frame(&mut self) -> FrameResult {
        let frame_time = self
            .last_frame_start
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        self.last_frame_start = None;
        self.frames_rendered += 1;

        // Record frame time
        self.frame_times.push_back(frame_time);
        while self.frame_times.len() > self.config.history_size {
            self.frame_times.pop_front();
        }

        // Check for dropped frame
        let dropped = frame_time > self.min_frame_time_ms();
        if dropped {
            self.frames_dropped += 1;
        }

        // Determine if quality adjustment needed
        let quality_change = if self.config.adaptive_quality {
            self.evaluate_quality_change()
        } else {
            QualityChange::None
        };

        // Apply quality change
        match quality_change {
            QualityChange::Decrease => {
                self.current_quality = self.lower_quality(self.current_quality);
            }
            QualityChange::Increase => {
                self.current_quality = self.higher_quality(self.current_quality);
            }
            QualityChange::None => {}
        }

        FrameResult {
            frame_time_ms: frame_time,
            dropped,
            quality_change,
            current_quality: self.current_quality,
        }
    }

    /// Evaluate if quality should be changed
    fn evaluate_quality_change(&self) -> QualityChange {
        if self.frame_times.len() < 5 {
            return QualityChange::None;
        }

        let avg_frame_time = self.average_frame_time();
        let target = self.target_frame_time_ms();

        if avg_frame_time > target * self.config.adjustment_threshold {
            // We're over budget, decrease quality
            QualityChange::Decrease
        } else if avg_frame_time < target * 0.7 && self.current_quality != QualityTier::Cinematic {
            // We have headroom, consider increasing quality
            QualityChange::Increase
        } else {
            QualityChange::None
        }
    }

    /// Get lower quality tier
    fn lower_quality(&self, current: QualityTier) -> QualityTier {
        match current {
            QualityTier::Cinematic => QualityTier::Ultra,
            QualityTier::Ultra => QualityTier::High,
            QualityTier::High => QualityTier::Medium,
            QualityTier::Medium => QualityTier::Low,
            QualityTier::Low => QualityTier::Minimal,
            QualityTier::Minimal => QualityTier::Minimal,
        }
    }

    /// Get higher quality tier
    fn higher_quality(&self, current: QualityTier) -> QualityTier {
        match current {
            QualityTier::Minimal => QualityTier::Low,
            QualityTier::Low => QualityTier::Medium,
            QualityTier::Medium => QualityTier::High,
            QualityTier::High => QualityTier::Ultra,
            QualityTier::Ultra => QualityTier::Cinematic,
            QualityTier::Cinematic => QualityTier::Cinematic,
        }
    }

    /// Get average frame time
    pub fn average_frame_time(&self) -> f64 {
        if self.frame_times.is_empty() {
            0.0
        } else {
            self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64
        }
    }

    /// Get current FPS
    pub fn current_fps(&self) -> f64 {
        let avg = self.average_frame_time();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    /// Get current quality tier
    pub fn current_quality(&self) -> QualityTier {
        self.current_quality
    }

    /// Set quality tier (disables adaptive for this frame)
    pub fn set_quality(&mut self, quality: QualityTier) {
        self.current_quality = quality;
    }

    /// Get total frames rendered
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Get total frames dropped
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Get drop rate
    pub fn drop_rate(&self) -> f64 {
        if self.frames_rendered == 0 {
            0.0
        } else {
            self.frames_dropped as f64 / self.frames_rendered as f64
        }
    }

    /// Should we skip this frame? (for frame limiting)
    pub fn should_render(&self) -> bool {
        match &self.last_frame_start {
            Some(start) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                elapsed >= self.target_frame_time_ms()
            }
            None => true,
        }
    }

    /// Time until next frame should be rendered
    pub fn time_until_next_frame(&self) -> Duration {
        match &self.last_frame_start {
            Some(start) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let target = self.target_frame_time_ms();
                if elapsed >= target {
                    Duration::ZERO
                } else {
                    Duration::from_secs_f64((target - elapsed) / 1000.0)
                }
            }
            None => Duration::ZERO,
        }
    }
}

/// Result of completing a frame
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Actual frame time in milliseconds
    pub frame_time_ms: f64,
    /// Was this frame dropped (over minimum acceptable time)?
    pub dropped: bool,
    /// Quality change that occurred
    pub quality_change: QualityChange,
    /// Current quality after any changes
    pub current_quality: QualityTier,
}

/// Quality change direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityChange {
    /// Quality decreased
    Decrease,
    /// Quality increased
    Increase,
    /// No change
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_basic() {
        let scheduler = RenderScheduler::new(SchedulerConfig::default());
        assert_eq!(scheduler.current_quality(), QualityTier::Medium);
        assert!((scheduler.target_frame_time_ms() - 16.67).abs() < 0.1);
    }

    #[test]
    fn test_frame_timing() {
        let mut scheduler = RenderScheduler::new(SchedulerConfig::default());

        scheduler.begin_frame();
        std::thread::sleep(Duration::from_millis(5));
        let result = scheduler.end_frame();

        assert!(result.frame_time_ms >= 5.0);
        assert!(!result.dropped);
        assert_eq!(scheduler.frames_rendered(), 1);
    }

    #[test]
    fn test_quality_change() {
        let mut scheduler = RenderScheduler::new(SchedulerConfig {
            target_fps: 60.0,
            adaptive_quality: true,
            history_size: 5,
            adjustment_threshold: 1.2,
            ..Default::default()
        });

        // Simulate slow frames
        for _ in 0..10 {
            scheduler.begin_frame();
            scheduler.frame_times.push_back(30.0); // 30ms frames (way over 16ms budget)
        }

        let _result = scheduler.end_frame();

        // Should have decreased quality
        assert!(scheduler.current_quality() < QualityTier::Medium);
    }

    #[test]
    fn test_fps_calculation() {
        let mut scheduler = RenderScheduler::new(SchedulerConfig::default());

        // Add some frame times
        scheduler.frame_times.push_back(16.67);
        scheduler.frame_times.push_back(16.67);
        scheduler.frame_times.push_back(16.67);

        let fps = scheduler.current_fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_drop_rate() {
        let mut scheduler = RenderScheduler::new(SchedulerConfig {
            min_fps: 30.0,
            ..Default::default()
        });

        scheduler.frames_rendered = 100;
        scheduler.frames_dropped = 10;

        assert!((scheduler.drop_rate() - 0.1).abs() < 0.001);
    }
}
