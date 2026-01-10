//! Query GPU compute limits for shared memory sizing.

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        let info = adapter.get_info();
        println!("=== GPU Info ===");
        println!("Name: {}", info.name);
        println!("Vendor: {:?}", info.vendor);
        println!("Backend: {:?}", info.backend);
        println!("Device type: {:?}", info.device_type);

        let limits = adapter.limits();
        println!("\n=== Compute Limits ===");
        println!("Max workgroup storage size: {} bytes ({} KB)",
            limits.max_compute_workgroup_storage_size,
            limits.max_compute_workgroup_storage_size / 1024);
        println!("Max workgroup size X: {}", limits.max_compute_workgroup_size_x);
        println!("Max workgroup size Y: {}", limits.max_compute_workgroup_size_y);
        println!("Max workgroup size Z: {}", limits.max_compute_workgroup_size_z);
        println!("Max workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
        println!("Max invocations per workgroup: {}", limits.max_compute_invocations_per_workgroup);

        println!("\n=== Memory Analysis for NTT ===");
        let shared_mem = limits.max_compute_workgroup_storage_size as usize;
        let bytes_per_element = 8; // u64
        let num_moduli = 3;

        let elements_single = shared_mem / bytes_per_element;
        let elements_fused = shared_mem / (bytes_per_element * num_moduli);

        println!("Elements per workgroup (single modulus): {}", elements_single);
        println!("Elements per workgroup (3 moduli fused): {}", elements_fused);

        // Calculate how many stages can be fused
        let log2 = |n: usize| (n as f64).log2() as usize;
        println!("\nFuseable NTT stages (single): {} (butterfly span up to {})",
            log2(elements_single), elements_single / 2);
        println!("Fuseable NTT stages (fused 3): {} (butterfly span up to {})",
            log2(elements_fused), elements_fused / 2);

        println!("\nFor n=8192 NTT (13 stages):");
        let stages_single = log2(elements_single).min(13);
        let stages_fused = log2(elements_fused).min(13);
        println!("  Single modulus: {} stages in shared mem, {} dispatches needed",
            stages_single, (13 + stages_single - 1) / stages_single);
        println!("  3 moduli fused: {} stages in shared mem, {} dispatches needed",
            stages_fused, (13 + stages_fused - 1) / stages_fused);
    });
}
