//! Server command

use catsith_api::{CatSithServer, ServerConfig};
use tracing::info;

pub async fn run(address: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting CatSith server on {}", address);

    let config = ServerConfig {
        listen_address: address.to_string(),
        ..Default::default()
    };

    let _server = CatSithServer::new(config);

    println!("CatSith Rendering Server");
    println!("========================");
    println!();
    println!("Listening on: {}", address);
    println!("Max clients:  {}", 64);
    println!();
    println!("Press Ctrl+C to stop");

    // In a real implementation, this would start a TCP/WebSocket server
    // For now, just wait
    println!();
    println!("(Server implementation pending - this is a placeholder)");

    // Keep running until interrupted
    tokio::signal::ctrl_c().await?;

    println!("\nShutting down...");

    Ok(())
}
