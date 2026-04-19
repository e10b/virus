#include "concurrent_buffers.h"

bool VoxelCompute::CircularBuffer::push(uint32_t value) {
	if (count >= BUFFER_SIZE) return false;
	buffer[tail] = value;
	tail = (tail + 1) & (BUFFER_SIZE - 1);
	count++;
	return true;
}

bool VoxelCompute::CircularBuffer::pop(uint32_t* value) {
	if (count == 0) return false;
	*value = buffer[head];
	head = (head + 1) & (BUFFER_SIZE - 1);
	count--;
	return true;
}

bool VoxelCompute::CircularBuffer::empty() const {
	return count == 0;
}

size_t VoxelCompute::CircularBuffer::size() const {
	return count;
}

void VoxelCompute::DoubleBuffer::init(size_t size) {
	buffer1 = new uint32_t[size];
	buffer2 = new uint32_t[size];
	currentWrite = buffer1;
	currentRead = buffer2;
}

void VoxelCompute::DoubleBuffer::swap() {
	if (useBuffer1) {
		currentWrite = buffer2;
		currentRead = buffer1;
	} else {
		currentWrite = buffer1;
		currentRead = buffer2;
	}
	useBuffer1 = !useBuffer1;
}

VoxelCompute::DoubleBuffer::~DoubleBuffer() {
	delete[] buffer1;
	delete[] buffer2;
}
