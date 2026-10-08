import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation
import Darwin

// Invariant: Layout is mirrored byte-for-byte in Python's _HEADER_FMT ('<4sIIIIIQIIiII12x').
// Any field change here requires a matching change in stt/system_audio.py.
public struct IPCHeader {
    var magic: UInt32 = 0x5245534F       // "RESO"
    var version: UInt32 = 2
    var sampleRate: UInt32 = 16000
    var channels: UInt32 = 1             // 1=System only, 2=System + Microphone
    var framesPerSlot: UInt32 = 4096
    var slotCount: UInt32 = 16
    var writeIndex: UInt64 = 0
    var command: UInt32 = 0              // 0=IDLE, 1=START_SYS, 2=STOP, 3=START_SYS_MIC
    var status: UInt32 = 0               // 0=IDLE, 1=STARTING, 2=CAPTURING, 3=FAILED
    var errorCode: Int32 = 0
    var requestID: UInt32 = 0            // Published by Python after command
    var responseID: UInt32 = 0           // Published by Swift after status/errorCode
    var padding: (UInt32, UInt32, UInt32) = (0, 0, 0)
}

class CaptureEngine: NSObject, SCStreamOutput, SCStreamDelegate {
    private var shmFd: Int32 = -1
    private var shmPointer: UnsafeMutableRawPointer!
    private let shmTotalSize: Int
    private let maxChannels = 2

    // Semaphore names must be ≤ 30 chars (Darwin PSHMNAMLEN kernel limit).
    private let shmPath     = "/tmp/res_audio_shm"
    private let dataSemName = "/res_aud_data"
    private let cmdSemName  = "/res_aud_cmd"

    private var dataSemaphore: UnsafeMutablePointer<sem_t>!
    private var cmdSemaphore: UnsafeMutablePointer<sem_t>!
    private var header: UnsafeMutablePointer<IPCHeader>!
    private var audioSlots: UnsafeMutablePointer<Float32>!

    private let stateQueue = DispatchQueue(label: "com.resonance.capture")
    private var activeRequestID: UInt32?
    private let audioLock = NSLock()
    private var internalBuffer: [Float32] = []
    private var micBuffer: [Float32] = []
    private var stream: SCStream?
    private var stopAudioTap: (() -> Void)?
    private var audioRequest: UInt32?
    private var audioEngine: AVAudioEngine?

    override init() {
        let headerSize = MemoryLayout<IPCHeader>.stride
        let dataSize   = Int(4096 * 16 * 2 * MemoryLayout<Float32>.stride)
        shmTotalSize   = headerSize + dataSize
        super.init()
        setupIPC()
        startCommandListener()
    }

    private func setupIPC() {
        let headerSize = MemoryLayout<IPCHeader>.stride

        // Use a plain file in /tmp for IPC instead of POSIX shm_open.
        // This avoids EACCES from stale root-owned POSIX SHM objects and
        // sidesteps the cdhash-based permission invalidation on unsigned builds.
        unlink(shmPath)
        sem_unlink(dataSemName)
        sem_unlink(cmdSemName)

        FileManager.default.createFile(atPath: shmPath, contents: Data(count: shmTotalSize))
        chmod(shmPath, 0o666)

        shmFd = Darwin.open(shmPath, O_RDWR)
        guard shmFd >= 0 else {
            fatalError("[Resonance] Failed to open SHM file: \(String(cString: strerror(errno)))")
        }
        guard ftruncate(shmFd, off_t(shmTotalSize)) == 0 else {
            fatalError("[Resonance] ftruncate failed: \(String(cString: strerror(errno)))")
        }

        shmPointer = mmap(nil, shmTotalSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        guard shmPointer != MAP_FAILED else {
            fatalError("[Resonance] mmap failed: \(String(cString: strerror(errno)))")
        }

        header     = shmPointer.bindMemory(to: IPCHeader.self, capacity: 1)
        header.pointee = IPCHeader()
        audioSlots = shmPointer.advanced(by: headerSize).bindMemory(to: Float32.self, capacity: 4096 * 16 * maxChannels)

        dataSemaphore = sem_open(dataSemName, O_CREAT | O_EXCL, 0o666, 0)
        cmdSemaphore  = sem_open(cmdSemName,  O_CREAT | O_EXCL, 0o666, 0)
        guard dataSemaphore != SEM_FAILED, cmdSemaphore != SEM_FAILED else {
            fatalError("[Resonance] sem_open failed: \(String(cString: strerror(errno)))")
        }

        if #available(macOS 14.2, *) {
            logNative("Native capture initialized: backend=Core Audio taps")
        } else {
            logNative("Native capture initialized: backend=ScreenCaptureKit")
        }
    }

    private func startCommandListener() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self else { return }
            while true {
                sem_wait(self.cmdSemaphore)
                self.stateQueue.async {
                    let request = self.header.pointee.requestID
                    guard request != self.activeRequestID else { return }
                    self.activeRequestID = request
                    switch self.header.pointee.command {
                    case 1: self.startCapture(includeMicrophone: false, request: request)
                    case 2:
                        self.stopResources()
                        self.respond(status: 0, request: request)
                        sem_post(self.dataSemaphore)
                        logNative("System capture stopped: request=\(request)")
                    case 3: self.startCapture(includeMicrophone: true, request: request)
                    default: break
                    }
                }
            }
        }
    }

    // Lifecycle methods and asynchronous completions run on stateQueue.
    // A newer request invalidates a callback immediately, before its command is handled.
    private func isCurrent(_ request: UInt32) -> Bool {
        activeRequestID == request && header.pointee.requestID == request
    }

    private func respond(status: UInt32, error: Int32 = 0, request: UInt32) {
        guard isCurrent(request) else { return }
        header.pointee.errorCode = error
        header.pointee.status = status
        header.pointee.responseID = request
    }

    private func failCapture(_ error: NSError, request: UInt32) {
        guard isCurrent(request) else { return }
        stopResources()
        respond(status: 3, error: Int32(error.code), request: request)
        sem_post(dataSemaphore)
        logNative("System capture failed: request=\(request); \(error)", level: "ERROR")
    }

    private func startCapture(includeMicrophone: Bool, request: UInt32) {
        guard isCurrent(request) else { return }
        stopResources()
        header.pointee.status = 1  // STARTING; responseID remains the previous response.
        header.pointee.errorCode = 0
        header.pointee.channels = includeMicrophone ? 2 : 1
        header.pointee.writeIndex = 0

        audioLock.lock()
        audioRequest = request
        audioLock.unlock()

        if includeMicrophone {
            // Permission dialogs must not block STOP or a newer capture request.
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                guard let self else { return }
                self.stateQueue.async {
                    guard self.isCurrent(request) else { return }
                    guard granted else {
                        self.failCapture(NSError(domain: "ResonanceCapture", code: -2,
                                                 userInfo: [NSLocalizedDescriptionKey: "Microphone permission required"]),
                                         request: request)
                        return
                    }
                    do {
                        try self.startMicrophoneCapture(request: request)
                        self.startSystemCapture(request: request)
                    } catch {
                        self.failCapture(NSError(domain: "ResonanceCapture", code: -3,
                                                 userInfo: [NSLocalizedDescriptionKey: "Microphone start failed",
                                                            NSUnderlyingErrorKey: error]), request: request)
                    }
                }
            }
        } else {
            startSystemCapture(request: request)
        }
    }

    private func startSystemCapture(request: UInt32) {
        // Selection is based only on API availability; errors never switch permission systems.
        if #available(macOS 14.2, *) {
            let tap = CoreAudioTap(onAudio: { [weak self] frames in
                guard let self else { return }
                self.stateQueue.async {
                    guard self.isCurrent(request) else { return }
                    self.audioLock.lock()
                    if self.audioRequest == request && self.header.pointee.status == 2 {
                        self.internalBuffer.append(contentsOf: frames)
                        self.processSlotsLocked()
                    }
                    self.audioLock.unlock()
                }
            }, onStarted: { [weak self] in
                guard let self else { return }
                self.stateQueue.async {
                    guard self.isCurrent(request) else { return }
                    // An idle tap may emit no buffers until another process plays audio.
                    self.respond(status: 2, request: request)
                    logNative("System capture started: request=\(request); channels=\(self.header.pointee.channels)")
                }
            }, onError: { [weak self] error in
                guard let self else { return }
                self.stateQueue.async { self.failCapture(error, request: request) }
            })
            stopAudioTap = { tap.stop() }
            tap.start()
            return
        }

        // Let ScreenCaptureKit request permission and report its own TCC error.
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { [weak self] content, error in
            guard let self else { return }
            self.stateQueue.async {
                guard self.isCurrent(request) else { return }
                if let error {
                    self.failCapture(error as NSError, request: request)
                    return
                }
                guard let display = content?.displays.first else {
                    self.failCapture(NSError(domain: "ResonanceCapture", code: -1,
                                             userInfo: [NSLocalizedDescriptionKey: "No displays found"]), request: request)
                    return
                }
                self.prepareStream(display: display, request: request)
            }
        }
    }

    private func prepareStream(display: SCDisplay, request: UInt32) {
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 1
        config.showsCursor = false

        let newStream = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            try newStream.addStreamOutput(self, type: .audio,
                                          sampleHandlerQueue: DispatchQueue(label: "com.resonance.audio",
                                                                            qos: .userInteractive))
        } catch {
            failCapture(error as NSError, request: request)
            return
        }
        audioLock.lock()
        stream = newStream
        audioLock.unlock()

        startStream(newStream, request: request)
    }

    private func startStream(_ newStream: SCStream, request: UInt32) {
        guard isCurrent(request) else { return }
        newStream.startCapture { [weak self] error in
            guard let self else { return }
            self.stateQueue.async {
                guard self.isCurrent(request) else {
                    // A cancelled start may still succeed inside ScreenCaptureKit.
                    if error == nil { newStream.stopCapture { _ in } }
                    return
                }
                if let error {
                    self.failCapture(error as NSError, request: request)
                } else {
                    self.respond(status: 2, request: request)
                    logNative("System capture started: request=\(request); channels=\(self.header.pointee.channels)")
                }
            }
        }
    }

    private func startMicrophoneCapture(request: UInt32) throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw NSError(domain: "ResonanceCapture", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "Microphone audio format unavailable"])
        }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let frameCapacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000.0 / inputFormat.sampleRate)
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: max(frameCapacity, 1024)) else { return }
            var error: NSError?
            var haveData = true
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                if haveData {
                    haveData = false
                    outStatus.pointee = .haveData
                    return buffer
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
            }
            if let floatChannelData = convertedBuffer.floatChannelData {
                let frames = UnsafeBufferPointer(start: floatChannelData[0], count: Int(convertedBuffer.frameLength))
                self.audioLock.lock()
                if self.audioRequest == request && self.header.pointee.requestID == request && self.header.pointee.status == 2 {
                    self.micBuffer.append(contentsOf: frames)
                    self.processSlotsLocked()
                }
                self.audioLock.unlock()
            }
        }

        audioEngine = engine  // Retain it before start so failure cleanup also removes the tap.
        try engine.start()
    }

    private func stopResources() {
        audioLock.lock()
        let previousStream = stream
        stream = nil
        audioRequest = nil
        internalBuffer.removeAll()
        micBuffer.removeAll()
        audioLock.unlock()
        stopAudioTap?()
        stopAudioTap = nil
        previousStream?.stopCapture { _ in }
        // Audio callbacks use audioLock; do not hold it while waiting for the engine to stop.
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        stateQueue.async {
            guard self.stream === stream, let request = self.activeRequestID else { return }
            self.failCapture(error as NSError, request: request)
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }

        var audioBufferList = AudioBufferList()
        var blockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: kCFAllocatorDefault, blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: 0, blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }

        withUnsafePointer(to: &audioBufferList.mBuffers) { buffersPtr in
            let buffers = UnsafeBufferPointer<AudioBuffer>(start: buffersPtr, count: Int(audioBufferList.mNumberBuffers))
            guard let mData = buffers[0].mData else { return }
            let frameCount = Int(buffers[0].mDataByteSize) / MemoryLayout<Float32>.size
            let frames = UnsafeBufferPointer(start: mData.bindMemory(to: Float32.self, capacity: frameCount),
                                             count: frameCount)
            audioLock.lock()
            if self.stream === stream && header.pointee.status == 2 &&
                header.pointee.requestID == header.pointee.responseID {
                internalBuffer.append(contentsOf: frames)
                processSlotsLocked()
            }
            audioLock.unlock()
        }
    }

    // Assumes: audioLock is held by caller.
    // Invariant: Interleaved 2-channel slot layout: [sys_0, mic_0, sys_1, mic_1, ...].
    // With microphone included, its clock keeps recording through system-audio silence.
    private func processSlotsLocked() {
        let framesPerSlot = Int(header.pointee.framesPerSlot)
        let channels = Int(header.pointee.channels)
        drainCaptureBuffers(system: &internalBuffer, microphone: &micBuffer,
                            channels: channels, framesPerSlot: framesPerSlot,
                            slotCount: Int(header.pointee.slotCount)) { slot in
            let slotIndex = Int(header.pointee.writeIndex % UInt64(header.pointee.slotCount))
            let destination = audioSlots.advanced(by: slotIndex * framesPerSlot * channels)
            destination.update(from: slot.baseAddress!, count: slot.count)
            header.pointee.writeIndex += 1
            sem_post(dataSemaphore)
        }
    }
}
