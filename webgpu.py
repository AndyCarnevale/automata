from typing import Any, Tuple
from abc import ABC, abstractmethod
import wgpu
import wgpu.backends.auto
import numpy as np
import pygame

from simulation import InitialStateRule

class GPUUpdateRule(ABC):
    @abstractmethod
    def update(self, current_state: np.ndarray) -> str:
        pass

class GPUConwayUpdate(GPUUpdateRule):
    def update(self, current_state: np.ndarray) -> str:
        pass

    def get_shader_code(self, grid_size) -> str:
        return f"""
        @group(0) @binding(0)
        var<storage, read> in_state: array<u32>;

        @group(0) @binding(1)
        var<storage, read_write> out_state: array<u32>;

        const N: u32 = {grid_size}u;

        fn wrap(a: i32, b: i32) -> i32 {{
            return ((a % b) + b) % b;
        }}

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {{
            let x = global_id.x;
            let y = global_id.y;
            if (x >= N || y >= N) {{
                return;
            }}
            let idx = x + y * N;

            var live_neighbors: u32 = 0u;
            for (var dy: i32 = -1; dy <= 1; dy++) {{
                for (var dx: i32 = -1; dx <= 1; dx++) {{
                    if (dx == 0 && dy == 0) {{
                        continue;
                    }}
                    let nx = u32(wrap(i32(x) + dx, i32(N)));
                    let ny = u32(wrap(i32(y) + dy, i32(N)));
                    let neighbor_idx = nx + ny * N;
                    live_neighbors += in_state[neighbor_idx];
                }}
            }}

            let cell_state = in_state[idx];
            var new_state: u32 = 0u;
            if (cell_state == 1u && (live_neighbors == 2u || live_neighbors == 3u)) {{
                new_state = 1u;
            }} else if (cell_state == 0u && live_neighbors == 3u) {{
                new_state = 1u;
            }}
            out_state[idx] = new_state;
        }}
        """

class GPUSimulationModel:
    def __init__(self, size: int, init_rule: InitialStateRule, sim_rule: GPUUpdateRule):
        self.size = size
        self.init_rule = init_rule
        self.sim_rule = sim_rule
        self.state = init_rule.rule(size)
        self.device = None
        self.buffer_a = None
        self.buffer_b = None
        self.staging_buffer = None
        self.compute_pipeline = None
        self.bind_group_a_to_b = None
        self.bind_group_b_to_a = None
    
    def initialize_webgpu(self):
        adapter = wgpu.gpu.request_adapter(canvas=None, power_preference="high-performance")
        self.device = adapter.request_device()
        
        buffer_size = self.state.nbytes
        
        self.buffer_a = self.device.create_buffer_with_data(
            data=self.state.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        )
        self.buffer_b = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        )
        
        self.staging_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        
        shader_module = self.device.create_shader_module(code=self.sim_rule.get_shader_code(self.size))
        bind_group_layout = self.device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ])
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
        self.compute_pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"}
        )
        
        self.bind_group_a_to_b = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.buffer_a}},
                {"binding": 1, "resource": {"buffer": self.buffer_b}},
            ],
        )
        self.bind_group_b_to_a = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.buffer_b}},
                {"binding": 1, "resource": {"buffer": self.buffer_a}},
            ],
        )

    def update(self, iteration):
        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.compute_pipeline)
        if iteration % 2 == 0:
            compute_pass.set_bind_group(0, self.bind_group_a_to_b, [], 0, 999999)
        else:
            compute_pass.set_bind_group(0, self.bind_group_b_to_a, [], 0, 999999)
        compute_pass.dispatch_workgroups(
            (self.size + 7) // 8,
            (self.size + 7) // 8
        )
        compute_pass.end()

        if iteration % 2 == 0:
            command_encoder.copy_buffer_to_buffer(self.buffer_b, 0, self.staging_buffer, 0, self.state.nbytes)
        else:
            command_encoder.copy_buffer_to_buffer(self.buffer_a, 0, self.staging_buffer, 0, self.state.nbytes)

        self.device.queue.submit([command_encoder.finish()])
        
        self.staging_buffer.map(mode=wgpu.MapMode.READ)
        buffer_view = self.staging_buffer.read_mapped()
        self.state = np.frombuffer(buffer_view, dtype=np.uint32).reshape(self.size, self.size)
        self.staging_buffer.unmap()

    def reset(self):
        new_state = self.init_rule.rule(self.size)
        self.device.queue.write_buffer(self.buffer_a, 0, new_state.tobytes())
        self.device.queue.write_buffer(self.buffer_b, 0, np.zeros_like(new_state).tobytes())
        self.state = new_state

class GPUSimulationView():
    def __init__(self, window_size: Tuple[int, int]):
        self.window_size = window_size
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Conway's Game of Life (WebGPU)")
        self.clock = pygame.time.Clock()

    def draw(self, state: np.ndarray):
        surf = pygame.surfarray.make_surface(state * 255)
        surf = pygame.transform.scale(surf, self.window_size)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

class GPUSimulationController():
    def __init__(self, model: GPUSimulationModel, view: GPUSimulationView) -> None:
        self.model = model
        self.view = view
        self.iteration = 0

    def initialize(self):
        self.model.initialize_webgpu()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.model.reset()
                    self.iteration = 0
        return True
    
    def update(self):
        self.model.update(self.iteration)
        self.view.draw(self.model.state)
        self.iteration += 1
        self.view.clock.tick(30)

    def close(self):
        pygame.quit()

class RandomInit(InitialStateRule):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def rule(self, size) -> np.ndarray:
        return np.random.randint(0, 2, size=(size * size), dtype=np.uint32)

def main():
    GRID_SIZE = 1000
    WINDOW_SIZE = (GRID_SIZE, GRID_SIZE)

    init_rule = RandomInit(p=0.75)
    sim_rule = GPUConwayUpdate()
    model = GPUSimulationModel(GRID_SIZE, init_rule, sim_rule)

    view = GPUSimulationView(WINDOW_SIZE)

    controller = GPUSimulationController(model, view)
    controller.initialize()

    is_running = True
    while is_running:
        is_running = controller.handle_events()
        controller.update()

    controller.close()

if __name__ == "__main__":
    main()
