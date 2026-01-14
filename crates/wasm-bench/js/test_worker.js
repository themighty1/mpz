// Test worker to check if WebGPU compute works in worker context
console.log('[test-worker] Test worker script loaded');

self.onmessage = async (e) => {
    console.log('[test-worker] Received message:', e.data);

    try {
        console.log('[test-worker] Checking WebGPU availability...');

        if (!navigator.gpu) {
            self.postMessage('ERROR: navigator.gpu not available in worker');
            return;
        }

        console.log('[test-worker] navigator.gpu exists, requesting adapter...');
        const adapter = await navigator.gpu.requestAdapter();

        if (!adapter) {
            self.postMessage('ERROR: Could not get GPU adapter');
            return;
        }

        console.log('[test-worker] Got GPU adapter, requesting device...');
        const device = await adapter.requestDevice();

        console.log('[test-worker] Got GPU device, running trivial compute job...');

        // Create a trivial compute shader that adds 1 to each element
        const shaderModule = device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read_write> data: array<u32>;

                @compute @workgroup_size(1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    data[global_id.x] = data[global_id.x] + 1u;
                }
            `
        });

        // Create input buffer with test data [1, 2, 3, 4]
        const inputData = new Uint32Array([1, 2, 3, 4]);
        const gpuBuffer = device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(gpuBuffer.getMappedRange()).set(inputData);
        gpuBuffer.unmap();

        // Create bind group
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            }]
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: gpuBuffer }
            }]
        });

        // Create pipeline
        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        const computePipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });

        // Execute compute
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(4);
        passEncoder.end();

        // Read back result
        const readBuffer = device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        commandEncoder.copyBufferToBuffer(gpuBuffer, 0, readBuffer, 0, inputData.byteLength);
        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readBuffer.getMappedRange());

        console.log('[test-worker] GPU compute result:', Array.from(result));
        self.postMessage('SUCCESS: WebGPU compute works! Input [1,2,3,4] -> Output [' + Array.from(result) + ']');

    } catch (error) {
        console.error('[test-worker] Error:', error);
        self.postMessage('ERROR: ' + error.toString());
    }
};

console.log('[test-worker] Message handler registered');
