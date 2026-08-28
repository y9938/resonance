import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation
import Darwin

// Invariant: Layout is mirrored byte-for-byte in Python's _HEADER_FMT ('<4sIIIIIQIIi20x').
// Any field change here requires a matching change in stt/system_audio.py.
public struct IPCHeader {
    var magic: UInt32 = 0x5245534F       // "RESO"
    var version: UInt32 = 1
    var sampleRate: UInt32 = 16000
    var channels: UInt32 = 1             // 1=System only, 2=System + Microphone
    var framesPerSlot: UInt32 = 4096
    var slotCount: UInt32 = 16
    var writeIndex: UInt64 = 0
    var command: UInt32 = 0              // 0=IDLE, 1=START_SYS, 2=STOP, 3=START_SYS_MIC
    var status: UInt32 = 0               // 0=IDLE, 1=STARTING, 2=CAPTURING, 3=FAILED
    var errorCode: Int32 = 0
    var padding: (UInt32, UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0, 0)
}

class CaptureEngine: NSObject, SCStreamOutput {
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

    private let audioLock = NSLock()
    private var internalBuffer: [Float32] = []
    private var micBuffer: [Float32] = []
    private var stream: SCStream?
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

        NSLog("[Resonance] IPC ready — %@", shmPath)
    }

    private func startCommandListener() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self else { return }
            while true {
                sem_wait(self.cmdSemaphore)
                switch self.header.pointee.command {
                case 1: self.startCapture(includeMicrophone: false)
                case 2: self.stopCapture()
                case 3: self.startCapture(includeMicrophone: true)
                default: break
                }
            }
        }
    }

    private func startCapture(includeMicrophone: Bool) {
        header.pointee.status    = 1  // STARTING
        header.pointee.errorCode = 0
        header.pointee.channels  = includeMicrophone ? 2 : 1

        // Do NOT gate on CGPreflightScreenCaptureAccess() — it always returns false
        // for ad-hoc / unsigned dev builds on macOS Sonoma/Sequoia (Cap issue #1722).
        // SCShareableContent will surface the TCC error (-3801) in its completionHandler.
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { [weak self] content, error in
            guard let self else { return }

            if let error {
                NSLog("[Resonance] SCShareableContent failed: %@", error.localizedDescription)
                self.header.pointee.status    = 3  // FAILED
                self.header.pointee.errorCode = Int32((error as NSError).code)
                self.header.pointee.command   = 0
                return
            }
            guard let display = content?.displays.first else {
                NSLog("[Resonance] No displays found")
                self.header.pointee.status    = 3
                self.header.pointee.errorCode = -1
                self.header.pointee.command   = 0
                return
            }

            let filter = SCContentFilter(display: display, excludingWindows: [])

            let config = SCStreamConfiguration()
            config.capturesAudio            = true
            config.excludesCurrentProcessAudio = true
            config.sampleRate               = 16000
            config.channelCount             = 1
            // Minimize GPU/CPU cost: SCK requires a video stream, so use the smallest
            // possible frame (2×2 px, 1 fps) to avoid rasterizing the full display.
            config.width                    = 2
            config.height                   = 2
            config.minimumFrameInterval     = CMTime(value: 1, timescale: 1)
            config.queueDepth               = 1
            config.showsCursor              = false

            let newStream = SCStream(filter: filter, configuration: config, delegate: nil)
            do {
                try newStream.addStreamOutput(self, type: .audio,
                                              sampleHandlerQueue: DispatchQueue(label: "com.resonance.audio",
                                                                                qos: .userInteractive))
            } catch {
                NSLog("[Resonance] addStreamOutput failed: %@", error.localizedDescription)
                self.header.pointee.status    = 3
                self.header.pointee.errorCode = Int32((error as NSError).code)
                self.header.pointee.command   = 0
                return
            }

            if includeMicrophone {
                self.startMicrophoneCapture()
            }

            newStream.startCapture { [weak self] error in
                guard let self else { return }
                if let error {
                    NSLog("[Resonance] startCapture failed: %@", error.localizedDescription)
                    self.header.pointee.status    = 3
                    self.header.pointee.errorCode = Int32((error as NSError).code)
                    self.header.pointee.command   = 0
                    self.stopMicrophoneCapture()
                } else {
                    self.stream = newStream
                    self.header.pointee.status  = 2  // CAPTURING
                    self.header.pointee.command = 0
                    NSLog("[Resonance] SCK System Audio Capture Started (channels: %d)", self.header.pointee.channels)
                }
            }
        }
    }

    private func startMicrophoneCapture() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            NSLog("[Resonance] Failed to create AVAudioConverter for microphone")
            return
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
                self.micBuffer.append(contentsOf: frames)
                self.processSlotsLocked()
                self.audioLock.unlock()
            }
        }

        do {
            try engine.start()
            self.audioEngine = engine
        } catch {
            NSLog("[Resonance] Failed to start AVAudioEngine: %@", error.localizedDescription)
        }
    }

    private func stopMicrophoneCapture() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        micBuffer.removeAll()
    }

    private func stopCapture() {
        audioLock.lock()
        stream?.stopCapture { _ in }
        stream = nil
        stopMicrophoneCapture()
        internalBuffer.removeAll()
        header.pointee.status  = 0  // IDLE
        header.pointee.command = 0
        audioLock.unlock()
        // Unblock Python reader that is blocked on data_sem.acquire().
        sem_post(dataSemaphore)
        NSLog("[Resonance] SCK System Audio Capture Stopped")
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
            internalBuffer.append(contentsOf: frames)
            processSlotsLocked()
            audioLock.unlock()
        }
    }

    // Assumes: audioLock is held by caller.
    // Invariant: Interleaved 2-channel slot layout: [sys_0, mic_0, sys_1, mic_1, ...].
    // Non-blocking sync: If micBuffer is behind sysChunk, pad remainder with zeros so system audio never stalls.
    private func processSlotsLocked() {
        let framesPerSlot = Int(header.pointee.framesPerSlot)
        let channels = Int(header.pointee.channels)

        if channels == 1 {
            while internalBuffer.count >= framesPerSlot {
                let slotIndex   = Int(header.pointee.writeIndex % UInt64(header.pointee.slotCount))
                let destination = audioSlots.advanced(by: slotIndex * framesPerSlot)
                internalBuffer.withUnsafeBufferPointer { src in
                    destination.update(from: src.baseAddress!, count: framesPerSlot)
                }
                internalBuffer.removeFirst(framesPerSlot)
                header.pointee.writeIndex += 1
                sem_post(dataSemaphore)
            }
        } else if channels == 2 {
            while internalBuffer.count >= framesPerSlot {
                let slotIndex   = Int(header.pointee.writeIndex % UInt64(header.pointee.slotCount))
                let destination = audioSlots.advanced(by: slotIndex * framesPerSlot * 2)

                let sysChunk = Array(internalBuffer.prefix(framesPerSlot))
                internalBuffer.removeFirst(framesPerSlot)

                var micChunk: [Float32]
                if micBuffer.count >= framesPerSlot {
                    micChunk = Array(micBuffer.prefix(framesPerSlot))
                    micBuffer.removeFirst(framesPerSlot)
                } else {
                    micChunk = Array(micBuffer)
                    micBuffer.removeAll()
                    if micChunk.count < framesPerSlot {
                        micChunk.append(contentsOf: [Float32](repeating: 0.0, count: framesPerSlot - micChunk.count))
                    }
                }

                for i in 0..<framesPerSlot {
                    destination[i * 2]     = sysChunk[i]
                    destination[i * 2 + 1] = micChunk[i]
                }

                header.pointee.writeIndex += 1
                sem_post(dataSemaphore)
            }
        }
    }
}
