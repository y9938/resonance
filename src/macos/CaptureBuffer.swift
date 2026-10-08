// Called under CaptureEngine's audio lock. Slots are mono SYS or interleaved SYS/MIC.
func drainCaptureBuffers(system: inout [Float], microphone: inout [Float],
                         channels: Int, framesPerSlot: Int, slotCount: Int,
                         publish: (UnsafeBufferPointer<Float>) -> Void) {
    let maxSamples = framesPerSlot * slotCount
    if system.count > maxSamples { system.removeFirst(system.count - maxSamples) }
    if microphone.count > maxSamples { microphone.removeFirst(microphone.count - maxSamples) }
    guard channels == 1 || channels == 2 else { return }

    // MIC provides the clock in dual mode: a silent system tap may emit no buffers.
    while (channels == 2 ? microphone.count : system.count) >= framesPerSlot {
        var slot = [Float](repeating: 0, count: framesPerSlot * channels)
        let available = min(system.count, framesPerSlot)
        for i in 0..<available { slot[i * channels] = system[i] }
        system.removeFirst(available)
        if channels == 2 {
            for i in 0..<framesPerSlot { slot[i * 2 + 1] = microphone[i] }
            microphone.removeFirst(framesPerSlot)
        }
        slot.withUnsafeBufferPointer(publish)
    }
}
