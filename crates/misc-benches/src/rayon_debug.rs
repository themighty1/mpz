//! Debug test for rayon parallelism in WASM.
//!
//! Tests whether rayon actually distributes work across threads.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use std::sync::Arc;

/// Debug test: create local rayon pool and run parallel work with logging.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn rayon_debug_test(concurrency: u32) -> String {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wasm_bindgen_futures::JsFuture;

    web_sys::console::log_1(&format!("[rayon_debug] Starting with concurrency={}", concurrency).into());

    let result: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let _handle = web_spawn::spawn(move || {
        web_sys::console::log_1(&"[rayon_debug] Worker started".into());

        // Create local pool
        web_sys::console::log_1(&format!("[rayon_debug] Building rayon pool with {} threads...", concurrency).into());

        let spawn_count = Arc::new(AtomicUsize::new(0));
        let spawn_count_clone = spawn_count.clone();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrency as usize)
            .spawn_handler(move |thread| {
                let count = spawn_count_clone.fetch_add(1, Ordering::SeqCst);
                web_sys::console::log_1(&format!("[rayon_debug] spawn_handler called, spawning thread #{}", count).into());
                let _ = web_spawn::spawn(move || {
                    web_sys::console::log_1(&format!("[rayon_debug] rayon thread #{} running", count).into());
                    thread.run();
                    web_sys::console::log_1(&format!("[rayon_debug] rayon thread #{} finished", count).into());
                });
                Ok(())
            })
            .build();

        let pool = match pool {
            Ok(p) => {
                web_sys::console::log_1(&format!("[rayon_debug] Pool created, spawn_handler called {} times", spawn_count.load(Ordering::SeqCst)).into());
                p
            }
            Err(e) => {
                let msg = format!("[rayon_debug] Pool creation failed: {}", e);
                web_sys::console::log_1(&msg.clone().into());
                *result_clone.lock().unwrap() = Some(msg);
                return;
            }
        };

        web_sys::console::log_1(&"[rayon_debug] Entering pool.install()...".into());

        let output = pool.install(|| {
            use rayon::prelude::*;

            web_sys::console::log_1(&"[rayon_debug] Inside pool.install()".into());

            // Simple parallel work: sum numbers
            let work_items: Vec<u64> = (0..1000).collect();

            web_sys::console::log_1(&"[rayon_debug] Starting par_iter...".into());

            let thread_usage = Arc::new(Mutex::new(std::collections::HashMap::<std::thread::ThreadId, usize>::new()));
            let thread_usage_clone = thread_usage.clone();

            let sum: u64 = work_items.par_iter().map(|&x| {
                // Track which thread processes this item
                let tid = std::thread::current().id();
                {
                    let mut usage = thread_usage_clone.lock().unwrap();
                    *usage.entry(tid).or_insert(0) += 1;
                }
                // Do some work
                (0..1000).fold(x, |acc, _| acc.wrapping_add(1))
            }).sum();

            let usage = thread_usage.lock().unwrap();
            let thread_count = usage.len();
            let distribution: Vec<_> = usage.values().collect();

            web_sys::console::log_1(&format!(
                "[rayon_debug] par_iter complete. sum={}, threads_used={}, distribution={:?}",
                sum, thread_count, distribution
            ).into());

            format!("sum={}, threads_used={}, distribution={:?}", sum, thread_count, distribution)
        });

        web_sys::console::log_1(&format!("[rayon_debug] pool.install() returned: {}", output).into());
        *result_clone.lock().unwrap() = Some(output);
    });

    // Poll for result
    loop {
        JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
            .await
            .unwrap();
        if let Some(r) = result.lock().unwrap().take() {
            return r;
        }
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }
}
