import AVFoundation
import CoreAudio
import Foundation

@available(macOS 14.2, *)
final class CoreAudioTap {
    // HAL calls can wait for a permission dialog. Keep them off the capture command queue.
    private let controlQueue = DispatchQueue(label: "com.resonance.audio-tap.control")
    private let audioQueue = DispatchQueue(label: "com.resonance.audio-tap.audio")
    private let lock = NSLock()
    private var cancelled = false
    private var started = false
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var deviceID = AudioObjectID(kAudioObjectUnknown)
    private var ioID: AudioDeviceIOProcID?
    private var formatListener: AudioObjectPropertyListenerBlock?
    private let tapUUID = UUID()
    private var inputFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                            channels: 1, interleaved: false)!
    private let onAudio: ([Float32]) -> Void
    private let onStarted: () -> Void
    private let onError: (NSError) -> Void

    init(onAudio: @escaping ([Float32]) -> Void, onStarted: @escaping () -> Void,
         onError: @escaping (NSError) -> Void) {
        self.onAudio = onAudio
        self.onStarted = onStarted
        self.onError = onError
    }

    func start() {
        controlQueue.async {
            do {
                try self.createTap()
                guard !self.isCancelled else { self.cleanup(); return }
                try self.createDevice()
                guard !self.isCancelled else { self.cleanup(); return }
                try self.startDevice()
                if self.isCancelled { self.cleanup() }
            } catch {
                self.cleanup()
                self.reportError(error as NSError)
            }
        }
    }

    func stop() {
        lock.lock()
        let wasCancelled = cancelled
        cancelled = true
        lock.unlock()
        if !wasCancelled { controlQueue.async { self.cleanup() } }
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    private func check(_ status: OSStatus, _ operation: String) throws {
        guard status == noErr else {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "Core Audio taps: \(operation) failed (\(status))"])
        }
    }

    private func createTap() throws {
        // Translate the native process ID to a HAL process object when it exists.
        var pid = getpid()
        var process = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                               UInt32(MemoryLayout<pid_t>.size), &pid, &size, &process)
        let excluded = status == noErr && process != kAudioObjectUnknown ? [process] : []
        let description = CATapDescription(monoGlobalTapButExcludeProcesses: excluded)
        description.name = "Resonance system audio"
        description.uuid = tapUUID
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tapID), "create tap")
        try updateFormat(tapID)
    }

    private func createDevice() throws {
        // A tap-only device avoids opening physical inputs or requiring microphone permission.
        let description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Resonance system audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: tapUUID.uuidString,
                                             kAudioSubTapDriftCompensationKey: true]]
        ]
        try check(AudioHardwareCreateAggregateDevice(description as CFDictionary, &deviceID), "create device")
    }

    private func startDevice() throws {
        let observedTap = tapID
        var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self, !self.isCancelled else { return }
            do { try self.updateFormat(observedTap) }
            catch { self.reportError(error as NSError) }
        }
        try check(AudioObjectAddPropertyListenerBlock(tapID, &address, audioQueue, listener), "observe format")
        formatListener = listener
        try check(AudioDeviceCreateIOProcIDWithBlock(&ioID, deviceID, audioQueue) { [weak self] _, input, _, _, _ in
            self?.readAudio(input)
        }, "create audio callback")
        guard !isCancelled else { return }
        try check(AudioDeviceStart(deviceID, ioID), "start device")
        lock.lock()
        started = true
        lock.unlock()
        if !isCancelled { onStarted() }
    }

    private func updateFormat(_ tap: AudioObjectID) throws {
        var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        var description = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &description), "read audio format")
        guard description.mSampleRate > 0, description.mChannelsPerFrame == 1,
              let format = AVAudioFormat(streamDescription: &description),
              let newConverter = AVAudioConverter(from: format, to: outputFormat) else {
            throw formatError("System audio format unavailable")
        }
        inputFormat = format
        converter = newConverter
    }

    private func readAudio(_ input: UnsafePointer<AudioBufferList>) {
        lock.lock()
        let running = started && !cancelled
        lock.unlock()
        guard running, input.pointee.mNumberBuffers == 1,
              let format = inputFormat, let converter,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: input, deallocator: nil),
              buffer.frameLength > 0 else { return }
        let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * 16000 / format.sampleRate)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, state in
            if supplied {
                state.pointee = .noDataNow
                return nil
            }
            supplied = true
            state.pointee = .haveData
            return buffer
        }
        if status == .error {
            reportError(error ?? formatError("System audio conversion failed"))
        } else if output.frameLength > 0, let samples = output.floatChannelData?[0] {
            onAudio(Array(UnsafeBufferPointer(start: samples, count: Int(output.frameLength))))
        }
    }

    private func formatError(_ message: String) -> NSError {
        NSError(domain: "ResonanceCoreAudioTap", code: -4,
                userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func reportError(_ error: NSError) {
        guard !isCancelled else { return }
        stop()
        onError(error)
    }

    // Only controlQueue owns the HAL handles. Audio callbacks never stop the device themselves.
    private func cleanup() {
        if let ioID {
            AudioDeviceStop(deviceID, ioID)
            AudioDeviceDestroyIOProcID(deviceID, ioID)
            self.ioID = nil
        }
        if let listener = formatListener {
            var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                                    mScope: kAudioObjectPropertyScopeGlobal,
                                                    mElement: kAudioObjectPropertyElementMain)
            AudioObjectRemovePropertyListenerBlock(tapID, &address, audioQueue, listener)
            formatListener = nil
        }
        if deviceID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(deviceID)
            deviceID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}
