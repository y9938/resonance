import Foundation
import ScreenCaptureKit
import CoreMedia
import Darwin

// Invariant: Layout is mirrored byte-for-byte in Python's _HEADER_FMT ('<4sIIIIIQIIi20x').
// Any field change here requires a matching change in stt/system_audio.py.
public struct IPCHeader {
    var magic: UInt32 = 0x5245534F       // "RESO"
    var version: UInt32 = 1
    var sampleRate: UInt32 = 16000
    var channels: UInt32 = 1
    var framesPerSlot: UInt32 = 4096
    var slotCount: UInt32 = 16
    var writeIndex: UInt64 = 0
    var command: UInt32 = 0              // 0=IDLE, 1=START, 2=STOP
    var status: UInt32 = 0               // 0=IDLE, 1=STARTING, 2=CAPTURING, 3=FAILED
    var errorCode: Int32 = 0
    var padding: (UInt32, UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0, 0)
}

class CaptureEngine: NSObject, SCStreamOutput {
    private var shmFd: Int32 = -1
    private var shmPointer: UnsafeMutableRawPointer!
    private let shmTotalSize: Int

    // Semaphore names must be ≤ 30 chars (Darwin PSHMNAMLEN kernel limit).
    private let shmPath     = "/tmp/res_audio_shm"
    private let dataSemName = "/res_aud_data"
    private let cmdSemName  = "/res_aud_cmd"

    private var dataSemaphore: UnsafeMutablePointer<sem_t>!
    private var cmdSemaphore: UnsafeMutablePointer<sem_t>!
    private var header: UnsafeMutablePointer<IPCHeader>!
    private var audioSlots: UnsafeMutablePointer<Float32>!

    private var internalBuffer: [Float32] = []
    private var stream: SCStream?

    override init() {
        let headerSize = MemoryLayout<IPCHeader>.stride
        let dataSize   = Int(4096 * 16 * MemoryLayout<Float32>.stride)
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
        audioSlots = shmPointer.advanced(by: headerSize).bindMemory(to: Float32.self, capacity: 4096 * 16)

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
                case 1: self.startCapture()
                case 2: self.stopCapture()
                default: break
                }
            }
        }
    }

    private func startCapture() {
        header.pointee.status    = 1  // STARTING
        header.pointee.errorCode = 0

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

            newStream.startCapture { [weak self] error in
                guard let self else { return }
                if let error {
                    NSLog("[Resonance] startCapture failed: %@", error.localizedDescription)
                    self.header.pointee.status    = 3
                    self.header.pointee.errorCode = Int32((error as NSError).code)
                    self.header.pointee.command   = 0
                } else {
                    self.stream = newStream
                    self.header.pointee.status  = 2  // CAPTURING
                    self.header.pointee.command = 0
                    NSLog("[Resonance] SCK System Audio Capture Started")
                }
            }
        }
    }

    private func stopCapture() {
        stream?.stopCapture { _ in }
        stream = nil
        internalBuffer.removeAll()
        header.pointee.status  = 0  // IDLE
        header.pointee.command = 0
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
            internalBuffer.append(contentsOf: frames)
        }

        let framesPerSlot = Int(header.pointee.framesPerSlot)
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
    }
}
