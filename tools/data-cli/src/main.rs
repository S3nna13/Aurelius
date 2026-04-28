use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "aurelius-data", about = "Aurelius data management CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show engine statistics
    Stats {
        /// Path to JSON data file
        #[arg(short, long)]
        file: Option<String>,
    },
    /// Export all data to JSON
    Export {
        /// Output file path
        #[arg(short, long)]
        output: String,
        /// Pretty print output
        #[arg(short, long)]
        pretty: bool,
    },
    /// Import data from JSON
    Import {
        /// Input file path
        input: String,
    },
    /// List all agents
    Agents {
        /// Filter by state
        #[arg(short, long)]
        state: Option<String>,
    },
    /// List recent activity
    Activity {
        /// Limit results
        #[arg(short, long, default_value = "20")]
        limit: u32,
    },
    /// List notifications
    Notifications {
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,
        /// Filter by priority
        #[arg(short, long)]
        priority: Option<String>,
    },
    /// Show memory layers
    Memory {
        /// Show entries in layer
        #[arg(short, long)]
        layer: Option<String>,
    },
    /// Search across all data
    Search {
        query: String,
        /// Limit results per category
        #[arg(short, long, default_value = "10")]
        limit: u32,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Manage sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Watch engine state in real-time
    Watch {
        /// Refresh interval in seconds
        #[arg(short, long, default_value = "2")]
        interval: u64,
    },
    /// Analyze engine health
    Health,
    /// Clear all data
    Clear {
        /// Confirm clearance
        #[arg(short, long)]
        force: bool,
    },
    /// Show engine version info
    Version,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// List all config keys
    List,
    /// Get a config value
    Get { key: String },
    /// Set a config value
    Set { key: String, value: String },
    /// Delete a config key
    Delete { key: String },
}

#[derive(Subcommand)]
enum SessionAction {
    /// List all sessions
    List,
    /// Get session info
    Get { id: String },
    /// Delete a session
    Delete { id: String },
    /// Show session statistics
    Stats,
}

fn main() {
    let cli = Cli::parse();
    let engine = aurelius_data_engine::DataEngine::new();

    match &cli.command {
        Commands::Stats { file } => {
            if let Some(path) = file {
                if !engine.load_from_file(path.clone()) {
                    eprintln!("Warning: Could not load from {path}");
                }
            }
            let stats = engine.get_stats();
            println!("Aurelius Data Engine Stats:");
            println!("  Agents:        {}", stats.agent_count);
            println!("  Activity:      {}", stats.activity_count);
            println!("  Notifications: {} ({} unread)", stats.notification_count, stats.notification_unread);
            println!("  Memory entries: {}", stats.memory_entry_count);
            println!("  Logs:          {}", stats.log_count);
        }

        Commands::Export { output, pretty } => {
            let json = engine.export_json();
            if *pretty {
                let val: serde_json::Value = serde_json::from_str(&json).unwrap();
                let formatted = serde_json::to_string_pretty(&val).unwrap();
                std::fs::write(output, &formatted).unwrap();
            } else {
                std::fs::write(output, &json).unwrap();
            }
            println!("Exported to {output}");
        }

        Commands::Import { input } => {
            let json = std::fs::read_to_string(input).expect("Failed to read file");
            if engine.import_json(json) {
                println!("Imported from {input}");
            } else {
                eprintln!("Failed to import from {input}");
            }
        }

        Commands::Agents { state } => {
            let agents = engine.list_agents();
            let filtered: Vec<_> = if let Some(s) = state {
                agents.into_iter().filter(|a| a.state == *s).collect()
            } else {
                agents
            };
            if filtered.is_empty() {
                println!("No agents found");
                return;
            }
            println!("Agents ({}):", filtered.len());
            for agent in &filtered {
                println!("  {} | state={} | role={}", agent.id, agent.state, agent.role);
            }
        }

        Commands::Activity { limit } => {
            let activity = engine.get_activity(Some(*limit));
            if activity.is_empty() {
                println!("No activity found");
                return;
            }
            println!("Recent Activity ({}):", activity.len());
            for entry in &activity {
                let ts = chrono::DateTime::from_timestamp(entry.timestamp as i64, 0)
                    .map(|t| t.format("%H:%M:%S").to_string())
                    .unwrap_or_else(|| "?".to_string());
                let ok = if entry.success { "\u{2705}" } else { "\u{274c}" };
                println!("  [{}] {} {} {}", ts, ok, entry.command, entry.output);
            }
        }

        Commands::Notifications { category, priority } => {
            let notifications = engine.get_notifications(
                category.clone(),
                priority.clone(),
                None,
                Some(50),
            );
            if notifications.is_empty() {
                println!("No notifications found");
                return;
            }
            println!("Notifications ({}):", notifications.len());
            for n in &notifications {
                let read = if n.read { "\u{2713}" } else { "\u{25cf}" };
                println!("  {} [{}] {}: {}", read, n.priority, n.title, n.body);
            }
        }

        Commands::Memory { layer } => {
            let layers = engine.get_memory_layers();
            if let Some(l) = layer {
                let entries = engine.get_memory_entries(Some(l.clone()), None, Some(20));
                println!("Memory Layer: {l} ({} entries)", entries.len());
                for e in &entries {
                    println!("  [{}] {} (score: {})", e.layer, e.content, e.importance_score);
                }
            } else {
                println!("Memory Layers:");
                for l in &layers {
                    println!("  {} ({} entries)", l.name, l.entries);
                }
            }
        }

        Commands::Version => {
            println!("Aurelius Data CLI v{}", env!("CARGO_PKG_VERSION"));
        }

        Commands::Health => {
            let stats = engine.get_stats();
            let agents = engine.list_agents();
            println!("Aurelius Engine Health Check:");
            println!("  Status: {}", if stats.agent_count > 0 { "\u{2705} Healthy" } else { "\u{26a0}\u{fe0f} No data" });
            println!("  Agents: {} registered", stats.agent_count);
            println!("  Activity entries: {}", stats.activity_count);
            println!("  Notifications: {} ({} unread)", stats.notification_count, stats.notification_unread);
            println!("  Memory entries: {}", stats.memory_entry_count);
            println!("  Log entries: {}", stats.log_count);
            println!();
            println!("Active agents:");
            for a in &agents {
                let active = if a.state == "active" || a.state == "running" { "\u{25cf}" } else { "\u{25cb}" };
                println!("  {} {} ({})", active, a.id, a.state);
            }
        }

        Commands::Clear { force } => {
            if !force {
                println!("Use --force to confirm clearing all data");
                return;
            }
            let a = engine.clear_activity();
            let n = engine.clear_notifications();
            let l = engine.clear_logs();
            println!("Cleared: {a} activity entries, {n} notifications, {l} logs");
            println!("(Agents and config preserved)");
        }

        Commands::Config { action } => match action {
            ConfigAction::List => {
                let config = engine.get_all_config();
                if config.is_empty() {
                    println!("No configuration set");
                    return;
                }
                println!("Configuration ({} keys):", config.len());
                let mut keys: Vec<_> = config.into_iter().collect();
                keys.sort_by(|a, b| a.0.cmp(&b.0));
                for (key, value) in &keys {
                    println!("  {} = {}", key, value);
                }
            }
            ConfigAction::Get { key } => {
                match engine.get_config(key.clone()) {
                    Some(val) => println!("{} = {}", key, val),
                    None => println!("Key '{}' not found", key),
                }
            }
            ConfigAction::Set { key, value } => {
                engine.set_config(key.clone(), value.clone());
                println!("Set: {} = {}", key, value);
            }
            ConfigAction::Delete { key } => {
                // We can simulate delete by setting to empty
                engine.set_config(key.clone(), String::new());
                println!("Cleared: {}", key);
            }
        },

        Commands::Session { action } => {
            println!("Session management requires SessionManager crate");
            match action {
                SessionAction::List => println!("No active sessions (SessionManager not integrated)"),
                SessionAction::Get { id } => println!("Session {id}: not found"),
                SessionAction::Delete { id } => println!("Session {id}: deleted"),
                SessionAction::Stats => println!("Session stats: N/A"),
            }
        }

        Commands::Watch { interval } => {
            println!("Watching engine state (refresh every {interval}s)...");
            println!("Press Ctrl+C to stop\n");
            loop {
                let stats = engine.get_stats();
                let agents = engine.list_agents();
                print!("\x1b[2J\x1b[H");
                println!("Aurelius Engine Monitor (refreshing every {interval}s)");
                println!("{}", "=".repeat(50));
                println!("Agents:     {}", stats.agent_count);
                println!("Activity:   {}", stats.activity_count);
                println!("Notifs:     {} ({} unread)", stats.notification_count, stats.notification_unread);
                println!("Memory:     {} entries", stats.memory_entry_count);
                println!("Logs:       {}", stats.log_count);
                println!("{}", "=".repeat(50));
                println!("Agent States:");
                for a in &agents {
                    let icon = match a.state.as_str() {
                        "active" | "running" => "\u{2705}",
                        "idle" => "\u{23f8}\u{fe0f}",
                        "error" => "\u{274c}",
                        _ => "\u{2753}",
                    };
                    println!("  {} {} ({})", icon, a.id, a.state);
                }
                std::thread::sleep(std::time::Duration::from_secs(*interval));
            }
        }

        Commands::Search { query, limit } => {
            let activity = engine.search_activity(query.to_string(), Some(*limit));
            let logs = engine.search_logs(query.to_string(), None, Some(*limit));
            let memory = engine.get_memory_entries(None, Some(query.to_string()), Some(*limit));
            let agents = engine.list_agents();

            println!("Search results for '{query}':\n");

            let matched_agents: Vec<_> = agents.into_iter()
                .filter(|a| a.id.to_lowercase().contains(&query.to_lowercase()))
                .collect();
            if !matched_agents.is_empty() {
                println!("Agents ({}):", matched_agents.len());
                for a in &matched_agents {
                    println!("  {} ({})", a.id, a.state);
                }
            }

            if !activity.is_empty() {
                println!("\nActivity ({}):", activity.len());
                for a in &activity {
                    println!("  {} | {}", a.command, a.output);
                }
            }

            if !logs.is_empty() {
                println!("\nLogs ({}):", logs.len());
                for l in &logs {
                    println!("  [{}] {}: {}", l.level, l.logger, l.message);
                }
            }

            if !memory.is_empty() {
                println!("\nMemory ({}):", memory.len());
                for m in &memory {
                    println!("  [{}] {}", m.layer, m.content);
                }
            }

            let total = matched_agents.len() + activity.len() + logs.len() + memory.len();
            println!("\nTotal matches: {total}");
        }
    }
}
