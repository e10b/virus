struct VoxelEditCommand {
    wordIndex: u32,
    wordValue: u32,
    _pad: u32,
    _pad2: u32,
};

@group(0) @binding(6) var<storage, read_write> brickPool: array<u32>;
@group(0) @binding(11) var<storage, read> editCount: array<u32>;
@group(0) @binding(12) var<storage, read> editCommands: array<VoxelEditCommand>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = editCount[0];
    let idx = gid.x;
    if (idx >= count) {
        return;
    }

    let cmd = editCommands[idx];
    brickPool[cmd.wordIndex] = cmd.wordValue;
}
