use std::{sync::{mpsc::Receiver, Arc}};
use anyhow::Result;
use crate::core::jarvis_signals::JarvisSignals;

pub fn main(signals: Arc<JarvisSignals>, feedback_rx: Receiver<String>) -> Result<()> {
    println!("Speech synthesis disabled - text responses only");
    
    while !signals.is_shutdown() {
        let text = match feedback_rx.recv() {
            std::result::Result::Ok(str) => str,
            Err(_) => break
        };

        // Instead of synthesizing speech, just print the response
        println!("🔊 VOICE OUTPUT: {}", text);
        
        // Simulate speaking time with a brief pause
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    Ok(())
}