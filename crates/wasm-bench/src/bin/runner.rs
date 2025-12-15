//! Chromiumoxide-based WASM benchmark runner.
//!
//! Launches headless Chrome and runs benchmarks.
//! Requires WASM to be built first with ./build-wasm.sh
//!
//! Usage:
//!   ./build-wasm.sh
//!   cargo run --release --bin wasm-bench-runner -- [OPTIONS]
//!
//! Options:
//!   --iterations <N>   Number of iterations per benchmark (default: 100)
//!   --samples <N>      Number of samples per benchmark (default: 10)
//!   --headed           Run with visible browser window

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::NavigateParams;
use futures::StreamExt;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::net::TcpListener;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BenchResult {
    name: String,
    iterations: u32,
    samples: u32,
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    per_iter_ms: f64,
    per_iter_us: f64,
    throughput: f64,
}

#[derive(Debug, Deserialize)]
struct BenchResults {
    garble: Option<Vec<BenchResult>>,
    evaluate: Option<Vec<BenchResult>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BenchOutput {
    results: BenchResults,
    formatted: String,
}

fn get_crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}


fn print_results(results: &BenchResults) {
    fn print_section(name: &str, benchmarks: &[BenchResult]) {
        println!("\n=== {} ===", name);
        println!(
            "{:<30} {:>12} {:>14} {:>18}",
            "Name", "Median (ms)", "Per-iter (us)", "Throughput (ops/s)"
        );
        println!("{}", "-".repeat(78));
        for b in benchmarks {
            println!(
                "{:<30} {:>12.2} {:>14.2} {:>18.1}",
                b.name, b.median_ms, b.per_iter_us, b.throughput
            );
        }
    }

    if let Some(ref garble) = results.garble {
        print_section("Garble (AES-128)", garble);
    }
    if let Some(ref evaluate) = results.evaluate {
        print_section("Evaluate (AES-128)", evaluate);
    }
}

/// Simple static file server
async fn serve_file(
    req: Request<hyper::body::Incoming>,
    base_dir: Arc<PathBuf>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let path = req.uri().path();
    let path = if path == "/" { "/index.html" } else { path };

    // Remove leading slash and resolve path
    let file_path = base_dir.join(path.trim_start_matches('/'));

    // Security: ensure path is within base_dir
    let file_path = match file_path.canonicalize() {
        Ok(p) if p.starts_with(base_dir.as_ref()) => p,
        _ => {
            return Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Full::new(Bytes::from("Not Found")))
                .unwrap());
        }
    };

    match tokio::fs::read(&file_path).await {
        Ok(contents) => {
            let mime = mime_guess::from_path(&file_path)
                .first_or_octet_stream()
                .to_string();

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", mime)
                .header("Cross-Origin-Opener-Policy", "same-origin")
                .header("Cross-Origin-Embedder-Policy", "require-corp")
                .body(Full::new(Bytes::from(contents)))
                .unwrap())
        }
        Err(_) => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap()),
    }
}

/// Start HTTP server and return the bound address
async fn start_server(base_dir: PathBuf) -> Result<SocketAddr, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let base_dir = Arc::new(base_dir);

    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => continue,
            };

            let base_dir = base_dir.clone();
            tokio::spawn(async move {
                let service =
                    service_fn(move |req| serve_file(req, base_dir.clone()));

                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });

    Ok(addr)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut iterations = 100u32;
    let mut samples = 10u32;
    let mut headless = true;

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                iterations = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(100);
            }
            "--samples" => {
                i += 1;
                samples = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--headed" => {
                headless = false;
            }
            "--help" | "-h" => {
                println!("WASM Benchmark Runner");
                println!();
                println!("Usage: wasm-bench-runner [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --iterations <N>   Number of iterations per benchmark (default: 100)");
                println!("  --samples <N>      Number of samples per benchmark (default: 10)");
                println!("  --headed           Run with visible browser window");
                println!("  --help, -h         Show this help");
                println!();
                println!("Note: Run ./build-wasm.sh first to build the WASM module.");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let crate_dir = get_crate_dir();
    let index_path = crate_dir.join("index.html");

    if !index_path.exists() {
        return Err(format!("index.html not found at {:?}", index_path).into());
    }

    let pkg_dir = crate_dir.join("pkg");
    if !pkg_dir.exists() {
        return Err("pkg/ directory not found. Run ./build-wasm.sh first to build the WASM module.".into());
    }

    // Start HTTP server
    let server_addr = start_server(crate_dir).await?;
    println!("Started HTTP server at http://{}", server_addr);

    println!("Launching browser (headless: {})...", headless);

    // Configure browser
    let mut builder = BrowserConfig::builder();
    if headless {
        builder = builder.arg("--headless=new");
    } else {
        // For headed mode, need to disable headless explicitly
        builder = builder.arg("--no-first-run").arg("--disable-gpu");
    }

    // Set window size
    builder = builder.window_size(1200, 800);

    let config = builder.build()?;
    println!("Browser config: {:?}", config);

    let (browser, mut handler) = Browser::launch(config).await?;

    // Spawn handler (filter out known false-positive error from chromiumoxide)
    // See: https://github.com/mattsse/chromiumoxide/issues/167
    let handle = tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if let Err(e) = event {
                if e.to_string() == "data did not match any variant of untagged enum Message" {
                    continue;
                }
                eprintln!("Browser event error: {:?}", e);
            }
        }
    });

    // Create page
    let page = browser.new_page("about:blank").await?;

    // Navigate to the benchmark page with autorun
    let url = format!(
        "http://{}/?autorun=true&iterations={}&samples={}",
        server_addr, iterations, samples
    );

    println!("Loading {}...", url);
    page.goto(NavigateParams::builder().url(&url).build()?)
        .await?;

    // Wait for navigation to complete
    page.wait_for_navigation().await?;
    println!("Page loaded, waiting for benchmarks...");

    // Wait for benchmarks to complete (poll for results)
    println!(
        "Running benchmarks ({} iterations, {} samples)...\n",
        iterations, samples
    );

    let timeout = Duration::from_secs(300); // 5 minute timeout
    let start = std::time::Instant::now();
    let mut last_status = String::new();

    let result: BenchOutput = loop {
        if start.elapsed() > timeout {
            return Err("Benchmark timed out after 5 minutes".into());
        }

        // Check for errors first
        let error_check = page
            .evaluate("window.__benchError || null")
            .await?;
        if let Ok(Some(error)) = error_check.into_value::<Option<String>>() {
            return Err(format!("JavaScript error: {}", error).into());
        }

        // Check progress
        let progress_check = page
            .evaluate("window.__benchProgress || null")
            .await?;
        if let Ok(Some(progress)) = progress_check.into_value::<Option<String>>() {
            if progress != last_status {
                // Clear line and print progress
                print!("\r\x1b[K{}", progress);
                use std::io::Write;
                std::io::stdout().flush().ok();
                last_status = progress;
            }
        }

        // Check if results are available
        let check = page
            .evaluate("window.__benchResults ? JSON.stringify(window.__benchResults) : null")
            .await?;

        match check.into_value::<Option<String>>() {
            Ok(Some(json_str)) => {
                println!(); // New line after progress
                break serde_json::from_str(&json_str)?;
            }
            Ok(None) => {}
            Err(_) => {}
        }

        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    // Print results
    print_results(&result.results);

    // Cleanup
    drop(page);
    drop(browser);
    handle.abort();

    println!("\nBenchmark complete.");
    Ok(())
}
